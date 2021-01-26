#   Intel(R) Single Event API
#
#   This file is provided under the BSD 3-Clause license.
#   Copyright (c) 2021, Intel Corporation
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#       Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#       Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#       Neither the name of the Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#   IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#********************************************************************************************************************************************************************************************************************************************************************************************

from __future__ import print_function
import os
import sys
import json
import time
import platform
import threading
from ctypes import cdll, c_char_p, c_void_p, c_ulonglong, c_int, c_double, c_long, c_bool, c_short, c_wchar_p, c_uint32, POINTER, CFUNCTYPE
from sea_runtool import reset_global, global_storage


class Dummy:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class Task:
    def __init__(self, itt, name, id, parent):
        self.itt = itt
        self.name = name
        self.id = id
        self.parent = parent

    def __enter__(self):
        self.itt.lib.itt_task_begin(self.itt.domain, self.id, self.parent, self.itt.get_string_id(self.name), 0)
        return self

    def arg(self, name, value):
        assert self.id
        try:
            value = float(value)
            self.itt.lib.itt_metadata_add(self.itt.domain, self.id, self.itt.get_string_id(name), value)
        except ValueError:
            self.itt.lib.itt_metadata_add_str(self.itt.domain, self.id, self.itt.get_string_id(name), str(value))
        return self

    def blob(self, name, pointer, size):
        assert self.id
        self.itt.lib.itt_metadata_add_blob(self.itt.domain, self.id, self.itt.get_string_id(name), pointer, size)
        return self

    def __exit__(self, type, value, traceback):
        self.itt.lib.itt_task_end(self.itt.domain, 0)
        return False


class Track:
    def __init__(self, itt, track):
        self.itt = itt
        self.track = track

    def __enter__(self):
        self.itt.lib.itt_set_track(self.track)
        return self

    def __exit__(self, type, value, traceback):
        self.itt.lib.itt_set_track(None)
        return False


def prepare_environ(args):
    if 'sea_env' in global_storage(None):
        return global_storage('sea_env')
    env = os.environ.copy()
    if args.verbose == 'info':
        env['INTEL_SEA_VERBOSE'] = '1'
    bitness = '32' if '32' in platform.architecture()[0] else '64'
    env_name = 'INTEL_LIBITTNOTIFY' + bitness
    if env_name not in env or 'SEAPI' not in env[env_name]:
        if sys.platform == 'win32':
            dl_name = 'IntelSEAPI%s.dll' % bitness
        elif sys.platform == 'darwin':
            dl_name = 'libIntelSEAPI.dylib'
        else:
            dl_name = 'libIntelSEAPI%s.so' % bitness

        env[env_name] = os.path.join(args.bindir, dl_name)
    if args.bindir not in env['PATH']:
        env['PATH'] += os.pathsep + args.bindir

    reset_global('sea_env', env)
    return global_storage('sea_env')


class ITT(object):
    scope_global = 1
    scope_process = 2
    scope_thread = 3
    scope_task = 4

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ITT, cls).__new__(cls)
        return cls._instance

    def __init__(self, domain):
        if hasattr(self, 'lib'):
            return
        bitness = 32 if '32' in platform.architecture()[0] else 64
        env_name = 'INTEL_LIBITTNOTIFY' + str(bitness)
        self.lib = None
        self.strings = {}
        self.tracks = {}
        self.counters = {}
        env = global_storage('sea_env')
        if env_name not in env:
            print("Warning:", env_name, "is not set...")
            return
        if os.path.exists(env[env_name]):
            self.lib = cdll.LoadLibrary(env[env_name])
        if not self.lib:
            print("Warning: Failed to load", env[env_name], "...")
            return

        # void* itt_create_domain(const char* str)
        self.lib.itt_create_domain.argtypes = [c_char_p]
        self.lib.itt_create_domain.restype = c_void_p

        # void* itt_create_string(const char* str)
        self.lib.itt_create_string.argtypes = [c_char_p]
        self.lib.itt_create_string.restype = c_void_p

        # void itt_marker(void* domain, uint64_t id, void* name, int scope)
        self.lib.itt_marker.argtypes = [c_void_p, c_ulonglong, c_void_p, c_int, c_ulonglong]

        # void itt_task_begin(void* domain, uint64_t id, uint64_t parent, void* name, uint64_t timestamp)
        self.lib.itt_task_begin.argtypes = [c_void_p, c_ulonglong, c_ulonglong, c_void_p, c_ulonglong]

        # void itt_task_begin_overlapped(void* domain, uint64_t id, uint64_t parent, void* name, uint64_t timestamp)
        self.lib.itt_task_begin_overlapped.argtypes = [c_void_p, c_ulonglong, c_ulonglong, c_void_p, c_ulonglong]

        # void itt_metadata_add(void* domain, uint64_t id, void* name, double value)
        self.lib.itt_metadata_add.argtypes = [c_void_p, c_ulonglong, c_void_p, c_double]

        # void itt_metadata_add_str(void* domain, uint64_t id, void* name, const char* value)
        self.lib.itt_metadata_add_str.argtypes = [c_void_p, c_ulonglong, c_void_p, c_char_p]

        # void itt_metadata_add_blob(void* domain, uint64_t id, void* name, const void* value, uint32_t size)
        self.lib.itt_metadata_add_blob.argtypes = [c_void_p, c_ulonglong, c_void_p, c_void_p, c_uint32]

        # void itt_task_end(void* domain, uint64_t timestamp)
        self.lib.itt_task_end.argtypes = [c_void_p, c_ulonglong]

        # void itt_task_end_overlapped(void* domain, uint64_t timestamp, uint64_t taskid)
        self.lib.itt_task_end_overlapped.argtypes = [c_void_p, c_ulonglong, c_ulonglong]

        # void* itt_counter_create(void* domain, void* name)
        self.lib.itt_counter_create.argtypes = [c_void_p, c_void_p]
        self.lib.itt_counter_create.restype = c_void_p

        # void itt_set_counter(void* id, double value, uint64_t timestamp)
        self.lib.itt_set_counter.argtypes = [c_void_p, c_double, c_ulonglong]

        # void* itt_create_track(const char* group, const char* track)
        self.lib.itt_create_track.argtypes = [c_char_p, c_char_p]
        self.lib.itt_create_track.restype = c_void_p

        # void itt_set_track(void* track)
        self.lib.itt_set_track.argtypes = [c_void_p]

        # uint64_t itt_get_timestamp()
        self.lib.itt_get_timestamp.restype = c_ulonglong

        if hasattr(self.lib, 'get_gpa_version'):
            # const char* get_gpa_version()
            self.lib.get_gpa_version.restype = c_char_p

        if sys.platform == 'win32':
            # long relog_etl(const char* szInput, const char* szOutput)
            self.lib.relog_etl.argtypes = [c_char_p, c_char_p]
            self.lib.relog_etl.restype = c_long
            # const char* resolve_pointer(const char* szModulePath, uint64_t addr)
            self.lib.resolve_pointer.argtypes = [c_char_p, c_ulonglong]
            self.lib.resolve_pointer.restype = c_char_p

            # bool ExportExeIconAsGif(LPCWSTR szExePath, LPCWSTR szGifPath)
            if hasattr(self.lib, 'ExportExeIconAsGif'):
                self.lib.ExportExeIconAsGif.argtypes = [c_wchar_p, c_wchar_p]
                self.lib.ExportExeIconAsGif.restype = c_bool

                # bool ConvertToGif(LPCWSTR szImagePath, LPCWSTR szGifPath, long width, long height)
                self.lib.ConvertToGif.argtypes = [c_wchar_p, c_wchar_p, c_long, c_long]
                self.lib.ConvertToGif.restype = c_bool

        elif 'linux' in sys.platform:
            # void itt_write_time_sync_markers()
            self.lib.itt_write_time_sync_markers.argtypes = []

        # typedef bool (*receive_t)(void* pReceiver, uint64_t time, uint16_t count, const wchar_t** names, const wchar_t** values, double progress);
        self.receive_t = CFUNCTYPE(c_bool, c_ulonglong, c_ulonglong, c_short, POINTER(c_wchar_p), POINTER(c_wchar_p), c_double)
        # typedef void* (*get_receiver_t)(const wchar_t* provider, const wchar_t* opcode, const wchar_t* taskName);
        self.get_receiver_t = CFUNCTYPE(c_ulonglong, c_wchar_p, c_wchar_p, c_wchar_p)
        if hasattr(self.lib, 'parse_standard_source'):
            # bool parse_standard_source(const char* file, get_receiver_t get_receiver, receive_t receive)
            self.lib.parse_standard_source.argtypes = [c_char_p, self.get_receiver_t, self.receive_t]
            self.lib.parse_standard_source.restype = c_bool

        self.domain = self.lib.itt_create_domain(domain.encode())

    def get_string_id(self, text):
        try:
            return self.strings[text]
        except:
            id = self.strings[text] = self.lib.itt_create_string(bytes(text, encoding='utf-8'))
            return id

    def marker(self, text, scope=scope_process, timestamp=0, id=0):
        if not self.lib:
            return
        self.lib.itt_marker(self.domain, id, self.get_string_id(text), scope, timestamp)

    def task(self, name, id=0, parent=0):
        if not self.lib:
            return Dummy()
        return Task(self, name, id, parent)

    def task_submit(self, name, timestamp, dur, id=0, parent=0):
        self.lib.itt_task_begin(self.domain, id, parent, self.get_string_id(name), timestamp)
        self.lib.itt_task_end(self.domain, timestamp + dur)

    def counter(self, name, value, timestamp=0):
        if not self.lib:
            return
        try:
            counter = self.counters[name]
        except:
            counter = self.counters[name] = self.lib.itt_counter_create(self.domain, self.get_string_id(name))
        self.lib.itt_set_counter(counter, value, timestamp)

    def track(self, group, name):
        if not self.lib:
            return Dummy()
        key = group + "/" + name
        try:
            track = self.tracks[key]
        except:
            track = self.tracks[key] = self.lib.itt_create_track(group, name)
        return Track(self, track)

    def get_timestamp(self):
        if not self.lib:
            return 0
        return self.lib.itt_get_timestamp()

    def relog(self, frm, to):
        if sys.platform == 'win32':
            if not self.lib:
                return
            self.lib.relog_etl(frm, to)

    def resolve_pointer(self, module, addr):
        if sys.platform == 'win32':
            if not self.lib:
                return
            return self.lib.resolve_pointer(module, addr)

    def time_sync(self):
        if not self.lib:
            return
        self.lib.itt_write_time_sync_markers()

    def parse_standard_source(self, path, reader):
        if not hasattr(self.lib, 'parse_standard_source'):
            return None
        receivers = []

        def receive(receiver, time, count, names, values, progress):  # typedef bool (*receive_t)(void* receiver, uint64_t time, uint16_t count, const wchar_t** names, const wchar_t** values, double progress);
            receiver = receivers[receiver - 1]  # Should be: receiver = cast(receiver, POINTER(py_object)).contents.value, but it doesn't work so we use index of the array
            args = {}
            for i in range(0, count):
                args[names[i]] = values[i]
            reader.set_progress(progress)
            receiver.receive(time, args)
            return True

        def get_receiver(provider, opcode, taskName):  # typedef void* (*get_receiver_t)(const wchar_t* provider, const wchar_t* opcode, const wchar_t* taskName);
            receiver = reader.get_receiver(provider, opcode, taskName)
            if not receiver:
                return 0
            receivers.append(receiver)
            return len(receivers)  # Should be: cast(pointer(py_object(receiver)), c_void_p).value, but it doesn't work, so we return index of the array

        return self.lib.parse_standard_source(bytes(path, encoding='utf-8'), self.get_receiver_t(get_receiver), self.receive_t(receive))

    def can_parse_standard_source(self):
        return hasattr(self.lib, 'parse_standard_source')

    def get_gpa_version(self):
        if not self.lib:
            return ""
        return self.lib.get_gpa_version()

    def export_exe_icon_as_gif(self, exe_path, gif_path):
        if sys.platform == 'win32':
            if not hasattr(self.lib, 'ExportExeIconAsGif'):
                return None
            return self.lib.ExportExeIconAsGif(exe_path, gif_path)
        elif sys.platform == 'darwin':
            import glob
            for file_path in glob.glob(os.path.normpath(os.path.join(exe_path, '../../Resources/*.icns'))):
                self.convert_image(file_path, gif_path, 16, 16)
                return True
            return False
        else:
            return False

    def convert_image(self, from_path, to_path, width, height):
        if sys.platform == 'win32':
            return self.lib.ConvertToGif(from_path, to_path, width, height)
        elif sys.platform == 'darwin':
            return os.system('sips -s format gif -z %d %d "%s" --out "%s"' % (width, height, from_path, to_path))
        else:
            os.system('convert %s -resize %dx%d %s' % (from_path, width, height, to_path))

def get_memory_usage(statics={}):  # in bytes
    if os.name == 'nt':
        if not statics:
            from ctypes import wintypes, Structure, c_size_t, sizeof, WinError, windll, byref

            class PROCESS_MEMORY_COUNTERS(Structure):
                _fields_ = [("cb", wintypes.DWORD),
                            ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", c_size_t),
                            ("WorkingSetSize", c_size_t),
                            ("QuotaPeakPagedPoolUsage", c_size_t),
                            ("QuotaPagedPoolUsage", c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", c_size_t),
                            ("QuotaNonPagedPoolUsage", c_size_t),
                            ("PagefileUsage", c_size_t),
                            ("PeakPagefileUsage", c_size_t)]

                def __init__(self):
                    self.cb = sizeof(self)

            windll.psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE, wintypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD)

            def get_working_set_size():
                pmi = PROCESS_MEMORY_COUNTERS()
                if not windll.psapi.GetProcessMemoryInfo(-1, byref(pmi), sizeof(pmi)):
                    raise WinError()
                return pmi.WorkingSetSize

            statics['GetWorkingSetSize'] = get_working_set_size

        return statics['GetWorkingSetSize']()
    else:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)  # kilobytes to bytes


def trace_execution(fn, args, save_to=None):
    import inspect
    if save_to:
        os.environ['INTEL_SEA_SAVE_TO'] = save_to
    itt = ITT("python")
    listener = None

    def extract_name(frame):
        name = frame.f_code.co_name
        if 'self' in frame.f_locals:
            cls = frame.f_locals['self'].__class__.__name__
            name = cls + "." + name
        return name

    if itt.lib:
        class ITTListener:
            def __init__(self):
                self.file_id = itt.get_string_id('__FILE__')
                self.line_id = itt.get_string_id('__LINE__')
                self.module_id = itt.get_string_id('__MODULE__')

            def begin(self, name, filename, lineno, mdl, args, task_id, frame):
                itt.lib.itt_task_begin_overlapped(itt.domain, task_id, 0, itt.get_string_id(name), 0)
                itt.lib.itt_metadata_add_str(itt.domain, task_id, self.file_id, filename)
                itt.lib.itt_metadata_add(itt.domain, task_id, self.line_id, lineno)
                if mdl:
                    itt.lib.itt_metadata_add_str(itt.domain, task_id, self.module_id, mdl)

            def end(self, task_id):
                itt.lib.itt_task_end_overlapped(itt.domain, 0, task_id)
                itt.counter('MEMORY_USAGE', get_memory_usage())

        listener = ITTListener(itt.lib)
    else:
        class Printer:
            def __init__(self):
                self.traced = set()

            def begin(self, name, filename, lineno, mdl, args, task_id, frame):
                if (filename, lineno) in self.traced:
                    return
                self.traced.add((filename, lineno))
                if '<' not in filename and any(isinstance(value, bytes) for value in args.values()):
                    print('%s:%d !BYTES! %s %s %s' % (filename, lineno, mdl, name, str(args)))
                    print('%s:%d !PARENT! %s %s' % (frame.f_back.f_code.co_filename, frame.f_back.f_code.co_firstlineno, extract_name(frame.f_back), str(frame.f_back.f_locals)))

            def end(self, task_id):
                pass

        listener = Printer()

    trace_execution.frames = {}
    trace_execution.recurrent = False
    high_part = 2**32

    def profiler(frame, event, arg):  # https://pymotw.com/2/sys/tracing.html
        if trace_execution.recurrent:
            return
        trace_execution.recurrent = True
        task_id = id(frame.f_code)
        if 'call' in event:
            if task_id in trace_execution.frames:
                trace_execution.frames[task_id] += 1
            else:
                trace_execution.frames[task_id] = 1
            task_id += trace_execution.frames[task_id] * high_part
            name = extract_name(frame)
            # print event, name, task_id, arg
            mdl = inspect.getmodule(frame)
            listener.begin(name, frame.f_code.co_filename, frame.f_code.co_firstlineno, mdl.__name__ if mdl else None, frame.f_locals, task_id, frame)
        elif 'return' in event:
            # print event, frame.f_code.co_name, task_id + trace_execution.frames[task_id] * high_part
            if task_id in trace_execution.frames:
                listener.end(task_id + trace_execution.frames[task_id] * high_part)
                if trace_execution.frames[task_id] > 1:
                    trace_execution.frames[task_id] -= 1
                else:
                    del trace_execution.frames[task_id]
        trace_execution.recurrent = False

    old_profiler = sys.getprofile()
    sys.setprofile(profiler)
    old_threading_profiler = threading.setprofile(profiler)
    if fn:
        fn(*args)
        sys.setprofile(old_profiler)
        threading.setprofile(old_threading_profiler)


def stack_task(itt):
    import random
    with itt.task("python_task"):
        time.sleep(0.01)
        if random.randrange(0, 2):
            stack_task(itt)
        time.sleep(0.01)


def test_itt():
    itt = ITT("python")
    itt.marker("Begin")
    ts1 = itt.get_timestamp()
    with itt.task("Main"):
        for i in range(0, 100):
            stack_task(itt)
            itt.counter("python_counter", i)
    ts2 = itt.get_timestamp()
    itt.marker("End")

    with itt.track("group", "track"):
        dur = (ts2 - ts1) / 100 + 1
        for ts in range(ts1, ts2, dur):
            itt.task_submit("submitted", ts, dur / 2)


def as_admin():
    if sys.platform != 'win32':
        return True

    from ctypes import windll

    def is_admin():
        try:
            return windll.shell32.IsUserAnAdmin()
        except:
            return False

    if is_admin():
        return True
    else:
        # Re-run the program with admin rights
        windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
    return False


def main():
    itt = ITT("python")
    return trace_execution(test_itt)

    class Reader:
        def set_progress(self, progress):
            pass

        def get_receiver(self, provider, opcode, taskName):
            class Receiver:
                def __init__(self, provider, opcode, taskName):
                    pass

                def receive(self, time, args):
                    pass

            return Receiver(provider, opcode, taskName)

    itt.parse_standard_source(r"etw-1.etl", Reader())


if __name__ == "__main__":
    main()
