"""
ert - Ensemble Reservoir Tool - a package for reservoir modeling.

The ert package itself has no code, but contains several subpackages:

ecl.ecl: Package for working with ECLIPSE files. The far most mature
   package in ert.

ecl.util:

The ert package is based on wrapping the libriaries from the ERT C
code with ctypes; an essential part of ctypes approach is to load the
shared libraries with the ctypes.CDLL() function. The ctypes.CDLL()
function uses the standard methods of the operating system,
i.e. standard locations configured with ld.so.conf and the environment
variable LD_LIBRARY_PATH.

To avoid conflict with other application using the ert libraries the
Python code should be able to locate the shared libraries without
(necessarily) using the LD_LIBRARY_PATH variable. The default
behaviour is to try to load from the library ../../lib64, but by using
the enviornment variable ERT_LIBRARY_PATH you can alter how ert looks
for shared libraries.

   1. By default the code will try to load the shared libraries from
      '../../lib64' relative to the location of this file.

   2. Depending on the value of ERT_LIBRARY_PATH two different
      behaviours can be imposed:

         Existing path: the package will look in the path pointed to
            by ERT_LIBRARY_PATH for shared libraries.

         Arbitrary value: the package will use standard load order for
         the operating system.

If the fixed path, given by the default ../../lib64 or ERT_LIBRARY_PATH
alternative fails, the loader will try the default load behaviour
before giving up completely.
"""
import os.path
import sys
import ctypes as ct

import warnings

warnings.filterwarnings(
    action="always",
    category=DeprecationWarning,
    module=r"ecl|ert",
)

from cwrap import Prototype

ECL_LEGACY_INSTALLED = not os.path.isdir(
    os.path.join(os.path.dirname(__file__), ".libs")
)
if not ECL_LEGACY_INSTALLED:
    from .version import version as __version__

    def get_include():
        return os.path.join(os.path.dirname(__file__), ".include")

    def dlopen_libecl():
        import ctypes
        import platform

        path = os.path.join(os.path.dirname(__file__), ".libs")
        if platform.system() == "Linux":
            path = os.path.join(path, "libecl.so")
        elif platform.system() == "Darwin":
            path = os.path.join(path, "libecl.dylib")
        elif platform.system() == "Windows":
            path = os.path.join(os.path.dirname(__file__), ".bin", "libecl.dll")
        else:
            raise NotImplementedError("Invalid platform")

        return ctypes.CDLL(path, ctypes.RTLD_GLOBAL)

    # Need to keep the function as a global variable so that we don't give C a
    # dangling pointer
    _abort_handler = None

    @ct.CFUNCTYPE(None, ct.c_char_p, ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_char_p)
    def _c_abort_handler(filename, lineno, function, message, backtrace):
        global _abort_handler
        if not _abort_handler:
            return
        _abort_handler(
            filename.decode(),
            lineno,
            function.decode(),
            message.decode(),
            backtrace.decode(),
        )

    def set_abort_handler(function):
        """
        Set callback function for util_abort, which is called prior to std::abort()
        """
        global _abort_handler
        _abort_handler = function

        EclPrototype.lib.util_set_abort_handler(_c_abort_handler)

    class EclPrototype(Prototype):
        lib = dlopen_libecl()

        def __init__(self, prototype, bind=True):
            super(EclPrototype, self).__init__(EclPrototype.lib, prototype, bind=bind)

else:
    #
    # If installed via CMake directly (legacy)
    #
    from cwrap import load as cwrapload

    try:
        import ert_site_init
    except ImportError:
        pass

    required_version_hex = 0x02070000

    ecl_lib_path = None
    ert_so_version = ""
    __version__ = "0.0.0"

    # 1. Try to load the __ecl_lib_info module; this module has been
    #    configured by cmake during the build configuration process. The
    #    module should contain the variable lib_path pointing to the
    #    directory with shared object files.
    try:
        from .__ecl_lib_info import EclLibInfo

        ecl_lib_path = EclLibInfo.lib_path
        ert_so_version = EclLibInfo.so_version
        __version__ = EclLibInfo.__version__
    except ImportError:
        pass
    except AttributeError:
        pass

    # 2. Using the environment variable ERT_LIBRARY_PATH it is possible to
    #    override the default algorithms. If the ERT_LIBRARY_PATH is set
    #    to a non existing directory a warning will go to stderr and the
    #    setting will be ignored.
    env_lib_path = os.getenv("ERT_LIBRARY_PATH")
    if env_lib_path:
        if os.path.isdir(env_lib_path):
            ert_lib_path = os.getenv("ERT_LIBRARY_PATH")
        else:
            sys.stderr.write(
                "Warning: Environment variable ERT_LIBRARY_PATH points to nonexisting directory:%s - ignored"
                % env_lib_path
            )

    # Check that the final ert_lib_path setting corresponds to an existing
    # directory.
    if ecl_lib_path:
        if not os.path.isabs(ecl_lib_path):
            ecl_lib_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), ecl_lib_path)
            )

        if not os.path.isdir(ecl_lib_path):
            ecl_lib_path = None

    if sys.hexversion < required_version_hex:
        raise Exception("ERT Python requires Python 2.7.")

    def load(name):
        try:
            return cwrapload(name, path=ecl_lib_path, so_version=ert_so_version)
        except ImportError:
            # For pip installs, setup.py puts the shared lib in this directory
            own_dir = os.path.dirname(os.path.abspath(__file__))
            return cwrapload(name, path=own_dir, so_version=ert_so_version)

    class EclPrototype(Prototype):
        lib = load("libecl")

        def __init__(self, prototype, bind=True):
            super(EclPrototype, self).__init__(EclPrototype.lib, prototype, bind=bind)


#
# Common
#

from .ecl_type import EclTypeEnum, EclDataType
from .ecl_util import (
    EclFileEnum,
    EclFileFlagEnum,
    EclPhaseEnum,
    EclUnitTypeEnum,
    EclUtil,
)

from .util.util import EclVersion
from .util.util import updateAbortSignals


updateAbortSignals()


def root():
    """
    Will print the filesystem root of the current ert package.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
