import sys
import os
import glob
import random
import pathlib
import socket
import ROOT
import XRootD.client

def is_zombie(file_path):
    # Try opening the ROOT file and check if it's a zombie file
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        print("WARNING! Found zombie file: {fp}".format(fp=file_path))
        return True
    f.Close()
    return False
                                

def buildFileListPosix(path):
    outfiles = []
    for root, dirs, fnames in os.walk(path):
        for fname in fnames:
            if fname.lower().endswith(".root"):
                outfiles.append(f"{root}/{fname}")
    return outfiles

def appendFilesXrd(filelist, xrdfs, path, suffixes = [".root"], recurse = False, num_clients = 16):
    status, dirlist = xrdfs.dirlist(path, flags = XRootD.client.flags.DirListFlags.STAT)

    if not status.ok:
        if status.code == 400 and status.errno == 3011:
            print("WARNING! XRootD directory not found: {p}".format(p=path))
        else:
            raise RuntimeError(f"Error in XRootD.client.FileSystem.dirlist: {status.message}, {status.code}, {status.errno}")
        return

    for diritem in dirlist:
        is_dir = diritem.statinfo.flags & XRootD.client.flags.StatInfoFlags.IS_DIR
        is_other = diritem.statinfo.flags & XRootD.client.flags.StatInfoFlags.OTHER
        is_file = not (is_dir or is_other)

        if is_dir and recurse:
            childpath = f"{path}/{diritem.name}"
            appendFilesXrd(filelist, xrdfs, childpath, suffixes=suffixes, recurse=recurse, num_clients=num_clients)
        elif is_file:
            lowername = diritem.name.lower()
            matchsuffix = False
            for suffix in suffixes:
                if lowername.endswith(suffix):
                    matchsuffix = True
                    break

            if matchsuffix:
                if num_clients > 0:
                    # construct client string if necessary to force multiple xrootd connections
                    # (needed for good performance when a single or small number of xrootd servers is used)
                    client = f"user_{random.randrange(num_clients)}"
                    outname = f"{xrdfs.url.protocol}://{client}@{xrdfs.url.hostname}:{xrdfs.url.port}/{path}/{diritem.name}"
                else:
                    outname = f"{xrdfs.url.protocol}://{xrdfs.url.hostid}/{path}/{diritem.name}"

                filelist.append(outname)


def buildFileListXrd(path, num_clients = 16):
    xrdurl =  XRootD.client.URL(path)

    if not xrdurl.is_valid():
        raise ValueError(f"Invalid xrootd path {path}")

    xrdfs = XRootD.client.FileSystem(xrdurl.hostid)
    xrdpath = xrdurl.path

    outfiles = []
    appendFilesXrd(outfiles, xrdfs, xrdpath, recurse=True, num_clients=num_clients)

    return outfiles


def buildFileList(path):
    xrdprefix = "root://"
    return buildFileListXrd(path) if path.startswith(xrdprefix) else buildFileListPosix(path)


def makeFilelist(paths, checkFileForZombie=False):
    filelist = []
    expandedPaths = []
    for path in paths:
        expandedPaths.append(path)
        print("Reading files from path {p}".format(p=path))
        files = buildFileList(path)
        filelist.extend(files)

    if checkFileForZombie:
        filelist = [p for p in paths if not is_zombie(p)]

    print("Length of list is {l} for paths {ep}".format(l=len(filelist), ep=expandedPaths))

    return filelist
