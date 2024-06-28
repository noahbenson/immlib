# -*- coding: utf-8 -*-
################################################################################
# immlib/pathlib/_osf.py


# Dependencies #################################################################

import mimetypes, json, os
from pathlib import (PosixPath, PurePosixPath, Path)
from urllib.parse import urlparse
from functools import reduce
from datetime import datetime

from cloudpathlib.cloudpath import (register_path_class, CloudPath, NoStatError)
from cloudpathlib.client import (register_client_class, Client)
from pcollections import (pdict, ldict, lazy)

from ..doc  import docwrap
from ..util import (is_str, is_amap, is_url, url_download)

# Utility Functions ############################################################

osf_basepath = 'https://api.osf.io/v2/nodes/%s/files/%s/'
osf_pagesize_format = 'page[size]='
osf_pagesize_check = 'page%5Bsize%5D='
osf_pagecache_filename = 'osf_treecache.json'
def _osf_pageload(proj, path,
                  pageno=0,
                  url=None,
                  storage='osfstorage',
                  cache_path=None,
                  mkdir_mode=0o775,
                  pagesize=100):
    """Loads a single page of the OSF contents for a specific OSF path."""
    cache_path = None if cache_path is None else Path(cache_path)
    cache_root = cache_path
    if url is None:
        path = '' if path is None else str(path).lstrip('/')
        cache_path = (cache_root / path) if cache_root else None
        url = (osf_basepath % (proj, storage)) + path
    if pagesize is not None:
        if osf_pagesize_format not in url and osf_pagesize_check not in url:
            if '?page=' in url:
                url = url + '&' + osf_pagesize_format + str(pagesize)
            elif osf_pagesize_check not in url:
                url = url + '?' + osf_pagesize_format + str(pagesize)
    else:
        pagesize = 0
    # First step is to load the data url.
    dat = None
    fromcache = False
    if cache_path is not None:
        # See if the JSON is in the cache.
        cache_flnm = f'.p{pageno}_{pagesize}.' + osf_pagecache_filename
        cache_flnm = cache_path / cache_flnm
        if cache_flnm.is_file():
            try:
                with cache_flnm.open('rt') as fl:
                    dat = json.load(fl)
                fromcache = True
            except Exception:
                pass
    if dat is None:
        # We need to load the data from the OSF website.
        dat = json.loads(url_download(url))
        fromcache = False
    # If there's no 'data' entry, we don't know what to do with it.
    if 'data' not in dat:
        raise ValueError(f'cannot detect kind of url entry for path:'
                         f' osf://{proj}/{path}')
    is_file = is_amap(dat['data'])
    if not (is_file or fromcache or cache_path is None):
        # This is a directory that we didn't load from cache, but
        # we have a cache path so we should save the url data. First,
        # make sure the directory exists.
        cache_flnm.parent.mkdir(mode=mkdir_mode, parents=True, exist_ok=True)
        # Now write the file.
        with cache_flnm.open('wt') as fl:
            json.dump(dat, fl)
    # At this point, the page has been loaded and cached; just return it.
    return (dat, cache_path)
def _osf_cache_file(url, path, mkdir_mode=0o775):
    """Downloads the given URL to the given path then returns the path."""
    # If there's no cache path, there's nothing to do here.
    if path is None:
        return None
    # Make sure the path exists.
    path = Path(path)
    if path.is_file():
        return path
    if not path.parent.is_dir():
        path.parent.mkdir(mode=mkdir_mode, parents=True)
    # Download the file and save it.
    url_download(url, destpath=path, mkdir_mode=mkdir_mode)
    if not path.is_file():
        raise RuntimeError(f"url failed to download: {url} -> {path}")
    return path
def _osf_fileentry(name, json, cache_path=None, mkdir_mode=0o775):
    if cache_path is None:
        cp = None
    else:
        cp = lazy(_osf_cache_file,
                  json['links']['download'], cache_path/name,
                  mkdir_mode=mkdir_mode)
    # Extract some meta-data.
    attrs = json['attributes']
    return ldict(
        kind='file',
        download_url=json['links']['download'],
        cache_path=cp,
        size=attrs.get('size'),
        date_modified=attrs.get('date_modified'),
        date_created=attrs.get('date_created'))
def _osf_crawl(proj, path=None,
               storage='osfstorage',
               cache_path=None,
               mkdir_mode=0o775,
               pagesize=100,
               url=None):
    """Private implementation of the osf_contents function."""
    if cache_path is None:
        cache_root = None
    else:
        cache_path = Path(cache_path)/proj/storage
        cache_root = cache_path
    # First step is to load the data url.
    (dat, cache_path) = _osf_pageload(proj, path=path, url=url,
                                      storage=storage,
                                      cache_path=cache_root,
                                      mkdir_mode=mkdir_mode,
                                      pagesize=pagesize)
    # Is this a file?
    if is_amap(dat['data']):
        raise RuntimeError("_osf_crawl given file instead of dir")
    # It's a directory, so we need to build up the list of directory contents.
    ls = {}
    pageno = 0
    while dat is not None:
        links = dat.get('links', {})
        dat = dat['data']
        for u in dat:
            r = u['attributes']
            rname = r['name']
            if r['kind'] == 'file':
                ls[rname] = _osf_fileentry(rname, u, cache_path, mkdir_mode)
            else:
                url = r['path'].lstrip('/')
                url = (osf_basepath % (proj, storage)) + url
                path = Path('/' if path is None else path)
                cp = None if cache_root is None else cache_root/rname
                ls[rname] = lazy(_osf_crawl,
                                 proj, path/rname,
                                 url=url,
                                 storage=storage,
                                 cache_path=cp,
                                 mkdir_mode=mkdir_mode,
                                 pagesize=pagesize)
        nxt = links.get('next')
        if nxt is None:
            dat = None
        else:
            pageno += 1
            (dat, cache_path) = _osf_pageload(proj, path=path, 
                                              pageno=pageno,
                                              url=nxt,
                                              storage=storage,
                                              cache_path=cache_root,
                                              mkdir_mode=mkdir_mode,
                                              pagesize=pagesize)
    return pdict(
        kind='directory',
        contents=ldict(ls),
        cache_path=cache_path)
@docwrap
def osf_contents(proj,
                 storage='osfstorage',
                 cache_path=None,
                 mkdir_mode=0o775,
                 pagesize=100,
                 lazy=True):
    """Returns a dictionary of the contents of the given OSF project and path.

    `osf_contents(project_name)` returns a dictionary of the contents of the OSF
    project with the given `project_name`. These contents are represented in a
    nested dictionary; each entry contains the keys `'kind'` (either `'file'` or
    `'directory'`), `'cache_path'` (the directory or filename of the associated
    cache), and either `'contents'` (for directories) or `'download_url'` (for
    files).

    Parameters
    ----------
    project : str
        The OSF project ID.
    storage : str, optional
        The OSF storage type to extract. By default this is `'osfstorage'`.
    cache_path : path or None, optional
        The cache directory in which to store the data downloaded.
    mkdir_mode : int, optional
        The mode to use when making directories in the cache. By default this is
        `0o775`.
    pagesize : int, optional
        The number of items to include in a single page when paging directory
        contents from the OSF server. The default is 100.
    lazy : bool, optional
        If `True` (the default), then the returned dictionary is a lazy dict
        representing the root of the project, but the OSF is not queried until
        its contents are requested. If `False`, then the directory contents are
        built immediately.

    Returns
    -------
    dict
        A nested lazy persistent dictionary of the project's contents.
    """
    if lazy:
        # Prepare the crawl data-structure, but don't build it yet.
        from pcollections import lazy
        crawl = lazy(_osf_crawl, proj,
                     storage=storage,
                     cache_path=cache_path,
                     mkdir_mode=mkdir_mode,
                     pagesize=pagesize)
        # Create a lazy-dict that just references crawl.
        return ldict(kind='directory',
                     contents=lazy(lambda:crawl()['contents']),
                     cache_path=lazy(lambda:crawl()['cache_path']))
    else:
        return _osf_crawl(proj,
                          storage=storage,
                          cache_path=cache_path,
                          mkdir_mode=mkdir_mode,
                          pagesize=pagesize)


# OSFClient ####################################################################

@register_client_class("osf")
class OSFClient(Client):
    """Client class for the OSF."""
    @staticmethod
    def _extract_path(contents, path):
        path = str(path)
        if path.startswith('osf://'):
            path = path[6:]
        path = path.lstrip('/')
        parts = PurePosixPath(path).parts[1:]
        for (ii,p) in enumerate(parts):
            if 'contents' not in contents:
                p = "/".join(parts[:ii+1])
                raise NotADirectoryError(f"Not a directory: {repr(p)}")
            contents = contents['contents']
            if p not in contents:
                p = "/".join(parts[:ii+1])
                raise FileNotFoundError(f"No such directory: {repr(p)}")
            contents = contents[p]
        return contents
    def __init__(self, project='xxxxx', storage='osfstorage',
                 file_cache_mode=None,
                 local_cache_dir=None,
                 content_type_method=mimetypes.guess_type,
                 pagesize=100,
                 mkdir_mode=0o775):
        super().__init__(
            file_cache_mode=file_cache_mode,
            local_cache_dir=local_cache_dir,
            content_type_method=content_type_method)
        # Currently, root must always be '/' (it's not in the options list).
        root = '/'
        self.project_id = project
        self.storage_provider = storage
        self.root_path = root
        self.mkdir_mode = mkdir_mode
        self.pagesize = pagesize
        self.file_cache_mode = file_cache_mode
        # Get the contents of the OSF repository (lazily).
        contents = osf_contents(
            project, 
            storage=storage,
            cache_path=local_cache_dir,
            pagesize=pagesize,
            mkdir_mode=mkdir_mode)
        # Save these as the whole-project contents.
        self.project_contents = contents
        # Now extract the root of these contents; if the root is '/' or '', then
        # we just use the project contents.
        root = str(root).lstrip('/')
        if root == '':
            self.root_contents = contents
        else:
            rc = lazy(self._extract_path, contents, root)
            self.root_contents = ldict(
                kind='directory',
                contents=lazy(lambda:rc()['contents']),
                cache_path=lazy(lambda:rc()['cache_path']))
    # Several of the abstract methods are non-operational for OSF, because all
    # OSF operations are currently read-only.
    def _move_file(self, src, dst, remove_src=True):
        raise RuntimeError(f"OSF CloudPath operations are read-only")
    def _remove(self, path, missing_ok=True):
        raise RuntimeError(f"OSF CloudPath operations are read-only")
    def _upload_file(self, local_path, cloud_path):
        raise RuntimeError(f"OSF CloudPath operations are read-only")
    # Other abstract methods are valid, however.
    def _download_file(self, cloud_path, local_path, mkdir_mode=0o775):
        if not isinstance(cloud_path, OSFPath):
            raise TypeError("cannot download path that is not an OSFPath")
        entry = self._extract_path(self.root_contents, cloud_path)
        local_path = Path(local_path)
        if entry['kind'] == 'directory':
            # Recursively download everything.
            for (name,ent) in entry['contents'].items():
                p = local_path/name
                if ent['kind'] == 'directory':
                    p.mkdir(mode=mkdir_mode, exist_ok=True)
                    self._download_file(cloud_path/name, p,
                                        mkdir_mode=mkdir_mode)
                else:
                    url_download(ent['download_url'], p)
        else:
            url_download(entry['download_url'], local_path)
        return local_path
    def _exists(self, cloud_path):
        if not isinstance(cloud_path, OSFPath):
            raise TypeError("cannot query path that is not an OSFPath")
        try:
            entry = self._extract_path(self.root_contents, cloud_path)
        except (NotADirectoryError, FileNotFoundError):
            return False
        return True
    def _list_dir(self, cloud_path, recursive=False):
        if not isinstance(cloud_path, OSFPath):
            raise TypeError("cannot list path that is not an OSFPath")
        if recursive:
            raise NotImplementedError(
                "recursive listing of OSF projects is not supported")
        entry = self._extract_path(self.root_contents, cloud_path)
        return ((cloud_path/k, v['kind'] == 'directory')
                for (k,v) in entry['contents'].items())
    def _path_kind(self, cloud_path):
        if not isinstance(cloud_path, OSFPath):
            raise TypeError("cannot query path that is not an OSFPath")
        entry = self._extract_path(self.root_contents, cloud_path)
        return entry['kind']
    def _path_entry(self, cloud_path):
        if not isinstance(cloud_path, OSFPath):
            raise TypeError("cannot query path that is not an OSFPath")
        return self._extract_path(self.root_contents, cloud_path)
    def _get_public_url(self, cloudpath):
        raise TypeError(
            f"{type(self)} does not support _generate_public_url")
    def _generate_presigned_url(self, cloudpath, expire_seconds=60*60):
        raise TypeError(
            f"{type(self)} does not support _generate_presigned_url")


# OSFPath ######################################################################

@register_path_class("osf")
class OSFPath(CloudPath):
    """Class for representing and operating on OSF repositories.

    `OSFPath(path)` returns a path object representing an OSF path; OSF paths
    use the format `osf://<project-id>/<path>`. The project ID is derived from
    the OSF tag; i.e., the website `https://osf.io/<project-id>` is the primary
    website of the OSF project. The project-ID may be followed by a colon and an
    OSF storage name (the project ID by itself is equivalent to
    `osf://<project-ID>:osfstorage/`).

    Parameters
    ----------
    cloud_path : str or path-like
        The OSF path that the created `OSFPath` object is to represent.
    client : OSFClient or None, optional
        The `OSFClient` object to use. The `OSFClient` is responsible primarily
        for the caching of data locally. If `OSFClient` is `None`, then an
        `OSFClient` object is created for the project using a temporary cache
        directory.
    local_cache_dir : str or path-like, optional
        The local directory in which cache files should be stored. This option
        is ignored if `client` is not `None`; otherwise it is passed to the
        created client object. The cache directory is the root cache directory
        for the entire OSF project.
    file_cache_mode : cloudpathlib.enums.FileCacheMode, optional
        How often to clear the file cache; see [cloudpathlib's caching
        docs](https://cloudpathlib.drivendata.org/stable/caching/) for more
        information about the options in `cloudpathlib.enums.FileCacheMode`.
    mkdir_mode : int, optional
        The mode to use when making directories in the cache. By default this is
        `0o775`. This option is ignored if the `client` option is not `None`.
    pagesize : int, optional
        The number of items to include in a single page when paging directory
        contents from the OSF server. The default is 100. This option is ignored
        if the `client` option is not `None`.
    """
    cloud_prefix = "osf://"
    client = OSFClient
    init_default_options = dict(
        local_cache_dir=None,
        file_cache_mode=None,
        mkdir_mode=0o775,
        pagesize=100)
    def __init__(self, cloud_path, client=None,
                 local_cache_dir=Ellipsis,
                 file_cache_mode=Ellipsis,
                 mkdir_mode=Ellipsis,
                 pagesize=Ellipsis):
        # Needed at the top of the init, see the CloudPath __init__ method in
        # cloudpath.py in cloudpathlib.
        self._handle = None
        self.client = OSFClient.get_default_client()
        # First validate the path.
        if isinstance(cloud_path, OSFPath):
            if client is None:
                client = clout_path.client
        else:
            # Go ahead and validate the url.
            self.is_valid_cloudpath(cloud_path, raise_on_error=True)
        # We'll also need to know the project and storage to create any new
        # client object. To do that we parse the cloud path URL.
        url = urlparse(str(cloud_path))
        if url.scheme != 'osf' or not url.netloc:
            raise ValueError(f"invalid OSF url: {repr(cloud_path)}")
        if ':' in url.netloc:
            (project, storage) = url.netloc.split(':')
        else:
            (project, storage) = (url.netloc, 'osfstorage')
        # Now that we have the project and storage, we can figure out the client
        # option, which we may be updating with options.
        if client is None:
            # No client was implied or given, so we make a new one from the path
            # and the remaining options. Any Ellipsis options we look up in the
            # default options dict (above).
            if local_cache_dir is Ellipsis:
                local_cache_dir = self.init_default_options['local_cache_dir']
            if file_cache_mode is Ellipsis:
                file_cache_mode = self.init_default_options['file_cache_mode']
            if mkdir_mode is Ellipsis:
                mkdir_mode = self.init_default_options['mkdir_mode']
            if pagesize is Ellipsis:
                pagesize = self.init_default_options['pagesize']
            client = OSFClient(project,
                               storage=storage,
                               local_cache_dir=local_cache_dir,
                               file_cache_mode=file_cache_mode,
                               mkdir_mode=mkdir_mode,
                               pagesize=pagesize)
        elif isinstance(client, OSFClient):
            # We can use the given client and just update any options whose
            # values aren't Ellipsis.
            change = False
            if local_cache_dir is Ellipsis:
                local_cache_dir = client._local_cache_dir
            elif local_cache_dir != client._local_cache_dir:
                change = True
            if file_cache_mode is Ellipsis:
                file_cache_mode = client.file_cache_mode
            elif file_cache_mode != client.file_cache_mode:
                change = True
            if mkdir_mode is Ellipsis:
                mkdir_mode = client.mkdir_mode
            elif mkdir_mode != client.mkdir_mode:
                change = True
            if pagesize is Ellipsis:
                pagesize = client.pagesize
            elif pagesize != client.pagesize:
                change = True
            # Make sure the path's project and storage match the client.
            if client.project_id != project: change = True
            if client.storage_provider != storage: change = True
            # If there's any change requested, we make a new client.
            if change:
                client = OSFClient(project,
                                   storage=storage,
                                   local_cache_dir=local_cache_dir,
                                   file_cache_mode=file_cache_mode,
                                   mkdir_mode=mkdir_mode,
                                   pagesize=pagesize)
        else:
            self.client = OSFClient()
            raise TypeError("OSFPaths require OSFClient objects as clients")
        # At this point we have a cloud path and client that are both valid.
        super().__init__(cloud_path, client=client)
    @property
    def bucket(self):
        return self.client.project_id
    @property
    def drive(self):
        return self._no_prefix.split("/", 1)[0]
    def is_dir(self):
        return self.client._path_kind(self) == "directory"
    def is_file(self):
        return self.client._path_kind(self) == "file"
    def mkdir(self, parents=False, exist_ok=False):
        raise TypeError(f"OSF CloudPath operations are read-only")
    def touch(self, exist_ok: bool = True):
        raise TypeError(f"OSF CloudPath operations are read-only")
    def stat(self):
        ent = self.client._path_entry(self)
        if ent['kind'] != 'file':
            raise NoStatError(f"No stats available for directory: {self}")
        mtime = ent.get('date_modified')
        ctime = ent.get('date_created')
        mtime = datetime.fromisoformat(mtime).timestamp() if mtime else 0
        # The [:-1] chops off a character at the end not recognized by the ISO
        # standard in all versions.
        ctime = datetime.fromisoformat(ctime[:-1]).timestamp() if ctime else 0
        stat = (
            None, # mode
            None, # ino
            self.cloud_prefix, # dev,
            None, # nlink,
            None, # uid,
            None, # gid,
            ent.get("size", 0), # size,
            None, # atime,
            mtime, # mtime,
            ctime) # ctime
        return os.stat_result(stat)
    @property
    def project_id(self):
        return self.client.project_id
    @property
    def _local(self):
        lcd = self.client._local_cache_dir
        base = lcd / self.project_id / self.client.storage_provider
        return base / self.key.lstrip('/')
    @property
    def key(self):
        return self._no_prefix_no_drive
