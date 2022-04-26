import os
import sys
import six
import json
import boto3
import shlex
import shutil
import tarfile
import tempfile
import contextlib
import subprocess
import textwrap
import importlib
import multiprocessing
from urllib.parse import urlparse
from retrying import retry
from subprocess import CalledProcessError
from sagemaker_inference import model_server
from custom_inference import handler


import logging

logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.INFO)

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)        

SAGEMAKER_BASE_PATH = os.path.join("/opt", "ml") 

file_name = os.path.basename(os.environ['SAGEMAKER_PROGRAM'])
module_name = os.path.splitext(file_name)[0]
code_dir = os.path.join(SAGEMAKER_BASE_PATH, "code")
module_dir = os.path.join(SAGEMAKER_BASE_PATH, "code")
model_dir = os.path.join(SAGEMAKER_BASE_PATH, "model")

os.environ['SAGEMAKER_MODEL_SERVER_WORKERS'] = str(multiprocessing.cpu_count())

for variable in os.environ:
    logger.debug(f'{variable}={os.environ[variable]}')

logger.debug(f'PYTHONPATH:{sys.path}')

HANDLER_SERVICE = handler.__name__

class ClientError(Exception):
    """Error class used to separate framework and user errors."""

class _CalledProcessError(ClientError):
    """This exception is raised when a process run by check_call() or
    check_output() returns a non-zero exit status.
    Attributes:
      cmd, return_code, output
    """

    def __init__(self, cmd, return_code=None, output=None):
        self.return_code = return_code
        self.cmd = cmd
        self.output = output
        super(_CalledProcessError, self).__init__()

    def __str__(self):
        if six.PY3 and self.output:
            error_msg = "\n%s" % self.output.decode("latin1")
        elif self.output:
            error_msg = "\n%s" % self.output
        else:
            error_msg = ""

        message = '%s:\nCommand "%s"%s' % (type(self).__name__, self.cmd, error_msg)
        return message.strip()

class InstallModuleError(_CalledProcessError):
    """Error class indicating a module failed to install."""

@contextlib.contextmanager
def tmpdir(suffix="", prefix="tmp", directory=None):  # type: (str, str, str) -> None
    """Create a temporary directory with a context manager. The file is deleted when the
    context exits.
    The prefix, suffix, and dir arguments are the same as for mkstemp().
    Args:
        suffix (str):  If suffix is specified, the file name will end with that suffix,
                       otherwise there will be no suffix.
        prefix (str):  If prefix is specified, the file name will begin with that prefix;
                       otherwise, a default prefix is used.
        directory (str):  If directory is specified, the file will be created in that directory;
                    otherwise, a default directory is used.
    Returns:
        str: path to the directory
    """
    tmp = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=directory)
    yield tmp
    shutil.rmtree(tmp)

def s3_download(url, dst):  # type: (str, str) -> None
    """Download a file from S3.
    Args:
        url (str): the s3 url of the file.
        dst (str): the destination where the file will be saved.
    """
    url = urlparse(url)

    if url.scheme != "s3":
        raise ValueError("Expecting 's3' scheme, got: %s in %s" % (url.scheme, url))

    bucket, key = url.netloc, url.path.lstrip("/")

    region = os.environ["AWS_REGION"]
    endpoint_url = os.environ.get('S3_ENDPOINT_URL', None)
    s3 = boto3.resource("s3", region_name=region, endpoint_url=endpoint_url)

    s3.Bucket(bucket).download_file(key, dst)

def download_and_extract(uri, path):
    """Download, prepare and install a compressed tar file from S3 or local directory as
    an entry point.
    SageMaker Python SDK saves the user provided entry points as compressed tar files in S3
    Args:
        uri (str): the location of the entry point.
        path (bool): The path where the script will be installed. It will not download and
                     install the if the path already has the user entry point.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.listdir(path):
        with tmpdir() as tmp:
            if uri.startswith("s3://"):
                dst = os.path.join(tmp, "tar_file")
                s3_download(uri, dst)

                with tarfile.open(name=dst, mode="r:gz") as t:
                    t.extractall(path=path)

            elif os.path.isdir(uri):
                if uri == path:
                    return
                if os.path.exists(path):
                    shutil.rmtree(path)
                shutil.copytree(uri, path)
            elif tarfile.is_tarfile(uri):
                with tarfile.open(name=uri, mode="r:gz") as t:
                    t.extractall(path=path)
            else:
                shutil.copy2(uri, path)

def write_file(path, data, mode="w"):
    with open(path, mode) as f:
        f.write(data)

def prepare(path, name):
    setup_path = os.path.join(path, "setup.py")
    if not os.path.exists(setup_path):
        data = textwrap.dedent(
            """
        from setuptools import setup
        setup(packages=[''],
              name="%s",
              version='1.0.0',
              include_package_data=True)
        """
            % name
        )

        logger.info("Module %s does not provide a setup.py. \nGenerating setup.py" % name)

        write_file(setup_path, data)

        data = textwrap.dedent(
            """
        [wheel]
        universal = 1
        """
        )

        logger.info("Generating setup.cfg")

        write_file(os.path.join(path, "setup.cfg"), data)

        data = textwrap.dedent(
            """
        recursive-include . *
        recursive-exclude . __pycache__*
        recursive-exclude . *.pyc
        recursive-exclude . *.pyo
        """
        )

        logger.info("Generating MANIFEST.in")

        write_file(os.path.join(path, "MANIFEST.in"), data)
    return None

def python_executable():
    """Returns the real path for the Python executable, if it exists.
    Returns RuntimeError otherwise.
    Returns:
        (str): the real path of the current Python executable
    """
    if not sys.executable:
        raise RuntimeError("Failed to retrieve the real path for the Python executable binary")
    return sys.executable

def create(cmd, error_class, cwd=None, capture_error=False, **kwargs):
    """Placeholder docstring"""
    try:
        stderr = subprocess.PIPE if capture_error else None
        return subprocess.Popen(
            cmd, env=os.environ, cwd=cwd or code_dir, stderr=stderr, **kwargs
        )
    except Exception as e:  # pylint: disable=broad-except
        six.reraise(error_class, error_class(e), sys.exc_info()[2])

def check_error(cmd, error_class, capture_error=False, **kwargs):
    process = create(cmd, error_class, capture_error=capture_error, **kwargs)

    if capture_error:
        _, stderr = process.communicate()
        return_code = process.poll()
    else:
        stderr = None
        return_code = process.wait()

    if return_code:
        raise error_class(return_code=return_code, cmd=" ".join(cmd), output=stderr)
    return process

def install(path, capture_error=False):
    """Install a Python module in the executing Python environment.
    Args:
        path (str):  Real path location of the Python module.
        capture_error (bool): Default false. If True, the running process captures the
            stderr, and appends it to the returned Exception message in case of errors.
    """
    cmd = "%s -m pip install . " % python_executable()

    if os.path.exists(os.path.join(path, "requirements.txt")):
        cmd += "-r requirements.txt"

    logger.info("Installing module with the following command:\n%s", cmd)

    check_error(
        shlex.split(cmd), InstallModuleError, cwd=path, capture_error=capture_error
    )

def import_module(uri, module_name):
    download_and_extract(uri, code_dir)

    prepare(code_dir, module_name)

    install(code_dir)

    importlib.import_module(module_name)
    return None
    
def _import_module():   
    
    logger.info('Check if model was repacked')
        
    default_script = 'inference.py'    
    code_dir = os.path.join(model_dir,'code')
    
    if os.path.isdir(code_dir):
        if os.path.isfile(os.path.join(code_dir, default_script)):
            module_name = default_script.replace('.py','')
        else:         
            py_files = [file for file in os.listdir(code_dir) if file.endswith('.py')]

            if len(py_files)>0:
                # select the first file on the list
                module_name = py_files[0].replace('.py','')
            else:
                logger.error('Inference code not found')
                return
                
        logger.debug('Model was repacked')
        logger.debug(f'Module code dir = {code_dir}')
        logger.debug(f'Module name = {module_name}')
                
        
        import_module(code_dir, module_name)
       
        # setting variables to use the imported module for inference      
        os.environ['SAGEMAKER_PROGRAM'] = module_name
        os.environ['SAGEMAKER_SUBMIT_DIRECTORY'] = code_dir
        os.environ['SAGEMAKER_CONTAINER_LOG_LEVEL'] = '20'
        
    else:
        
        logger.info('Importing module provided')
        logger.debug(f'Module dir = {module_dir}')
        logger.debug(f'Module name = {module_name}')
        import_module(module_dir, module_name)
                           

def _retry_if_error(exception):
    return isinstance(exception, CalledProcessError or OSError)

@retry(stop_max_delay=1000 * 50,
       retry_on_exception=_retry_if_error)
def _start_mms():
    logger.info('Starting model server')
    model_server.start_model_server(handler_service=HANDLER_SERVICE)
    logger.info('Model server started')

def main():
    if sys.argv[1] == 'serve':
        _import_module()
        _start_mms()

    else:
        subprocess.check_call(shlex.split(' '.join(sys.argv[1:])))
        
main()