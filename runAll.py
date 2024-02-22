#!/usr/bin/env python3

# examples
#
# all working points with default options
# python runAll.py -i /scratch/shared/NanoAOD/Tnp_NanoV9/TNP/ -o testAll
#
# only mc, and only steps 1, 4, 6
# python runAll.py -i /scratch/shared/NanoAOD/Tnp_NanoV9/TNP/ -o testAll -r mc -s 1 4 6
#
# could use -m to merge all output files into a single one, but would also need to change histogram names
# because at the moment they are always the same, it is the file name that distinguishes the working points
#
# use -d to test the command, without running them automatically
#
# typical default command for all steps
# python runAll.py -i input -o output --noOppositeChargeTracking

import os, re, copy, math, array

import argparse
import sys

def safeSystem(cmd, dryRun=False, quitOnFail=True):
    print(cmd)
    if not dryRun:
        res = os.system(cmd)
        if res:
            print('-'*30)
            print("safeSystem(): error occurred when executing the following command. Aborting")
            print(cmd)
            print('-'*30)
            if quitOnFail:
                quit()
        return res
    else:
        return 0

workingPoints = { 1: "reco",
                  2: "tracking",
                  3: "idip",
                  4: "trigger",
                  5: "iso",
                  6: "isonotrig",
                  7: "veto"
}

    
if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--indir',  default=None, type=str, required=True,
                        help='Input directory with the root files inside (common path for data and MC)')
    parser.add_argument('-o','--outdir', default=None, type=str, required=True,
                        help='Output directory to store all root files')
    parser.add_argument('-d',  '--dryRun', action='store_true',
                        help='Do not execute commands, just print them')
    parser.add_argument('-r',  '--run', default="all", type=str, choices=["data", "mc", "stand", "bkg", "Ztautau", "TTSemileptonic", "ZZ", "WZ", "WW", "all"],
                        help='Choose what to run, either data or MC, or both')
    # FIXME: unless I change histogram names inside the files I can't merge different working points, I could just merge data with MC but not worth
    #parser.add_argument('-m',  '--merge', action='store_true',
    #                    help='Merge root files in a new one')
    parser.add_argument('-nw', '--noVertexPileupWeight', action='store_true',
                        help='Do not use weights for vertex z position')
    parser.add_argument("-nos", "--noOppositeCharge", action="store_true",
                        help="Don't require opposite charges between tag and probe (including tracking, unless also using --noOppositeChargeTracking)")
    parser.add_argument(        "--noOppositeChargeTracking", action="store_true",
                                help="Don't require opposite charges between tag and probe for tracking")
    parser.add_argument("-sc", "--SameCharge", action="store_true", help="Require the TP Pair to have same sign (for bkg study)",
                        default=False)
    parser.add_argument('-s','--steps', default=None, nargs='*', type=int, choices=list(workingPoints.keys()),
                        help='Default runs all working points, but can choose only some if needed')
    parser.add_argument('-x','--exclude', default=None, nargs='*', type=int, choices=list(workingPoints.keys()),
                        help='Default runs all working points, but can choose to skip some if needed')
    parser.add_argument('-wpc','--workinPointsByCharge', default=["reco", "tracking", "idip", "trigger"], nargs='*', type=str, choices=list(workingPoints.values()),
                        help='These steps will be made charge dependent')
    parser.add_argument("-trk", "--trackerMuons", action="store_true",
                        help="Use tracker muons and a different executable")
    parser.add_argument("-p","--eventParity", help="Select events with given parity for statistical tests, -1/1 for odd/even events, 0 for all (default)",
                        type=int, nargs='+', default=[0], choices=[-1, 0, 1])
    parser.add_argument("-tpt","--tagPt", help="Minimum pt to select tag muons",
                        type=float, default=25.)
    parser.add_argument("-tiso","--tagIso", help="Isolation threshold to select tag muons",
                        type=float, default=0.15)
    parser.add_argument(        "--standaloneValidHits", help="Minimum number of valid hits for the standalone track (>= this value)",
                        type=int, default=1)
    #parser.add_argument('-exe', '--executable', default="Steve.py", type=str, choices=["Steve.py", "Steve_tracker.py"],
    #                    help='Choose script to run')

    parser.add_argument("-y", "--year", help="Choose year to run",
                        type=str, default="2016", choices=["2016", "2017", "2018"])
    parser.add_argument("-iso","--isoDefinition",help="Choose between the old and new isolation definition, 0 is old, 1 is new", default=1, type=int, choices = [0,1])


    args = parser.parse_args()

    outdir = args.outdir
    if not outdir.endswith("/"):
        outdir += "/"
    indir = args.indir
    if not indir.endswith("/"):
        indir += "/"

    executable = "Steve_tracker.py" if args.trackerMuons else "Steve.py"
        
    if not os.path.exists(outdir):
        print(f"Creating folder {outdir}")
        safeSystem(f"mkdir -p {outdir}", dryRun=False)

    inputdir_dict = {"data":"SingleMuon/", "mc":"DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/", "Ztautau":"DYJetsToTauTau_M-50_AtLeastOneEorMuDecay_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/", "TTSemileptonic":"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/", "ZZ": "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/", "WZ":"WZ_TuneCP5_13TeV-pythia8/", "WW": "WW_TuneCP5_13TeV-pythia8/"}
    isBkg_dict = {"data": 0, "mc": 0, "Ztautau": 1, "TTSemileptonic": 1, "ZZ": 1, "WZ": 1, "WW": 1}

    #inputdir_data = "SingleMuon/"
    #inputdir_mc   = "DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/"
    #inputdir_Ztautau = "DYJetsToTauTau_M-50_AtLeastOneEorMuDecay_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/"
    #inputdir_TTSemileptonic = "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/"
    #inputdir_ZZ = "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/"
    #inputdir_WZ = "WZ_TuneCP5_13TeV-pythia8/"
    #inputdir_WW = "WW_TuneCP5_13TeV-pythia8/"   
    
    toRun = []
    if args.run in ["all", "data","stand"]:
        toRun.append("data")
    if args.run in ["all", "mc", "stand"]:
        toRun.append("mc")
    if args.run in ["all", "bkg", "Ztautau"]:
        toRun.append("Ztautau")
    if args.run in ["all", "bkg", "TTSemileptonic"]:
        toRun.append("TTSemileptonic")
    if args.run in ["all", "bkg", "ZZ"]:
        toRun.append("ZZ")
    if args.run in ["all", "bkg", "WZ"]:
        toRun.append("WZ")
    if args.run in ["all", "bkg", "WW"]:
        toRun.append("WW")

    outfiles = [] # store names of output files so to merge them if needed

    postfix = "vertexWeights{v}_oscharge{c}".format(v="0" if args.noVertexPileupWeight  else "1",
                                                    c="0" if args.noOppositeCharge      else "1")

    commonOption = f" --tagPt {args.tagPt} --tagIso {args.tagIso} --standaloneValidHits {args.standaloneValidHits}"

    for xrun in toRun:

        isdata = 1 if xrun == "data" else 0
        isBkg = isBkg_dict[xrun]
        #inpath = indir + (inputdir_data if isdata else inputdir_mc)
        inpath = indir + inputdir_dict[xrun]
        for wp in workingPoints.keys():            
            if args.exclude and wp in args.exclude:
                continue
            if args.steps and wp not in args.steps:
                continue
            charges = [-1, 1] if workingPoints[wp] in args.workinPointsByCharge else [0]
            for ch in charges:
                for parity in args.eventParity:
                    step = workingPoints[wp]
                    if ch:
                        step += "plus" if ch == 1 else "minus"
                    if parity:
                        step += "odd" if parity == -1 else "even"
                    if wp == 2 and args.noOppositeChargeTracking:
                        postfixTracking = postfix.replace("oscharge1", "oscharge0")
                        outfile = f"{outdir}tnp_{step}_{xrun}_{postfixTracking}.root"
                    else:
                        outfile = f"{outdir}tnp_{step}_{xrun}_{postfix}.root"
                    outfiles.append(outfile)
                    cmd = f"python {executable} -i {inpath} -o {outfile} -d {isdata} -b {isBkg} -e {wp} -c {ch} -p {parity} -y {args.year} -iso {args.isoDefinition}"
                    cmd += commonOption
                    if args.noVertexPileupWeight:
                        cmd += " -nw"
                    if args.noOppositeCharge:
                        cmd += " -nos"
                    if args.noOppositeChargeTracking and wp == 2:
                        cmd += " --noOppositeChargeTracking "
                    if args.SameCharge:
                        cmd += " --SameCharge"
                    print("")
                    eventParityText = "all" if parity == 0 else "odd" if parity < 0 else "even"
                    print(f"Running for {xrun} and {step} efficiency ({eventParityText} events)")
                    safeSystem(cmd, dryRun=args.dryRun)
                    print("")

    ## FIXME: implement the merging if useful, but it depends on the name convention for histograms
    ##
    # if args.merge:
    #     mergedFile = f"{outdir}tnp_all_{postfix}.root"
    #     sourcefiles = " ".join(outfiles)
    #     haddcmd = f"hadd -f {mergedFile} {sourcefiles}"
    #     print("")
    #     print(f"Merging root files with hadd")
    #     safeSystem(haddcmd, dryRun=args.dryRun)
    #     print("")
