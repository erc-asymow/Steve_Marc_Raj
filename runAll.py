#!/usr/bin/env python3

# examples
#
# all working points with default options
# python runAll.py -i /scratch/shared/NanoAOD/Tnp_NanoV9/TNP/ -o testAll
#
# only signal mc, and only steps 1, 4, 6
# python runAll.py -i /scratch/shared/NanoAOD/Tnp_NanoV9/TNP/ -o testAll -r mc -s 1 4 6
#
# could use -m to merge all output files into a single one, but would also need to change histogram names
# because at the moment they are always the same, it is the file name that distinguishes the working points
#
# use -d to test the command, without running them automatically
#
# typical default command for all steps
# python runAll.py -i input -o output

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
                  7: "isofailtrig",
                  8: "veto",
                  9: "trackingTrackSeededTrack",
                  10: "recoTrackerMuon",
                  11: "vetoTrackerOrGlobalMuon",
                 }

def common_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("-tpt","--tagPt", help="Minimum pt to select tag muons",
                        type=float, default=25.)
    parser.add_argument("-tiso","--tagIso", help="Isolation threshold to select tag muons",
                        type=float, default=0.15)
    parser.add_argument("--standaloneValidHits", help="Minimum number of valid hits for the standalone track (>= this value)",
                        type=int, default=1)
    parser.add_argument("-y", "--year", help="Choose year to run",
                        type=str, default="2016", choices=["2016", "2017", "2018"])
    parser.add_argument('-nw', '--noVertexPileupWeight', action='store_true',
                        help='Do not use weights for vertex z position')
    parser.add_argument("-nos", "--noOppositeCharge", action="store_true",
                        help="Don't require opposite charges between tag and probe (tracking is an exception, unless also using --oppositeChargeTracking)")
    parser.add_argument("--oppositeChargeTracking", dest="noOppositeChargeTracking", action="store_false",
                        help="Require opposite charges between tag and probe for tracking (default case does not require it)")
    parser.add_argument("-sc", "--SameCharge", action="store_true", help="Require the TP Pair to have same sign (for bkg study)",
                        default=False)

    parser.add_argument("-iso","--isoDefinition",help="Choose between the old and new isolation definition, 0 is old, 1 is new", default=1, type=int, choices = [0,1])
    parser.add_argument("--noNormalizeMCsumGenWeights", dest="normalizeMCsumGenWeights", action="store_false", help="Divide MC yields by sum of gen weigths (the sum is stored anyway so it can be done later offline)")
    parser.add_argument("--maxFiles", help="Maximum number of files, for tests (default is all)",
                        type=int, default=0)
    # customize pt ranges and muon/track pt selection
    parser.add_argument("--histMinPt", help="Minimum value for the histogram pt binning",
                        type=float, default=10)
    parser.add_argument("--innerTrackMinPt", help="Minimum value for the Track/Muon pt (should always be smaller than --histMinPt",
                        type=float, default=10)
    parser.add_argument("--standaloneMinPt", help="Minimum value for the standalone muon pt cut",
                        type=float, default=15)

    
    return parser
        
                  7: "veto"
}

inputdir_dict = { "data" : "SingleMuon/", 
                  "mc" : ["DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/",
                          "DYJetsToMuMu_H2ErratumFix_PDFExt_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/",
                          "DYJetsToMuMu_M-10to50_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/"],
                  "Ztautau" : "DYJetsToTauTau_M-50_AtLeastOneEorMuDecay_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/",
                  "TTFullyleptonic" : "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/",
                  "TTSemileptonic" : "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/",
                  "WplusJets" : "WplusJetsToMuNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/",
                  "WminusJets" : "WminusJetsToMuNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/",
                  "ZZ" : "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/", 
                  "WZ" : "WZ_TuneCP5_13TeV-pythia8/", 
                  "WW" : "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8" #"WW_TuneCP5_13TeV-pythia8/"
}
#inputdir_data = "SingleMuon/"
#inputdir_mc   = "DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/"
#inputdir_Ztautau = "DYJetsToTauTau_M-50_AtLeastOneEorMuDecay_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/"
#inputdir_TTSemileptonic = "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/"
#inputdir_ZZ = "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/"
#inputdir_WZ = "WZ_TuneCP5_13TeV-pythia8/"
#inputdir_WW = "WW_TuneCP5_13TeV-pythia8/"   

isBkg_dict = {"data": 0, "mc": 0, "Ztautau": 1, "TTSemileptonic": 1, "TTFullyleptonic": 1, "WplusJets": 1, "WminusJets":1, "ZZ": 1, "WZ": 1, "WW": 1}


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
if __name__ == "__main__":    

    lumiDict = {"2016" : 16.8, # only postVFP
                "2017" : 37.99, # this includes the LS where the HLT_isoMu24 was not prescaled
                "2018" : 59.81}

    pb2fb = 1000.0 # conversion of cross section from pb to fb

    BR_TAUToMU = 0.1739
    BR_TAUToE = 0.1782
    xsec_ZmmPostVFP = 2001.9
    xsec_WpmunuPostVFP = 11765.9
    xsec_WmmunuPostVFP = 8703.87
    xsec_ZmmMass10to50PostVFP = 6997.0
    Z_TAU_TO_LEP_RATIO = (1.-(1. - BR_TAUToMU - BR_TAUToE)**2)

    inputdir_dict = {"data": {"path" : "SingleMuon/",
                              "isBkg": 0,
                              "xsec" : 1.0}, # dummy, just for consistency with other processes
                     "mc": {"path" : ["DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/", "DYJetsToMuMu_H2ErratumFix_PDFExt_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"],
                            "isBkg" : 0,
                            "xsec" : xsec_ZmmPostVFP},
                     "DYlowMass": {"path" : ["DYJetsToMuMu_M-10to50_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos"], # "DYJetsToMuMu_M-10to50_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos_ext1"],
                                   "isBkg" : 1,
                                   "xsec" : xsec_ZmmMass10to50PostVFP},
                     "Ztautau": { "path" : "DYJetsToTauTau_M-50_AtLeastOneEorMuDecay_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/",
                                   "isBkg" : 1,
                                   "xsec" : xsec_ZmmPostVFP*Z_TAU_TO_LEP_RATIO},
                     "TTSemileptonic": {"path" : "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/",
                                        "isBkg" : 1,
                                        "xsec" : 88.29},
                     "ZZ": {"path" : "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/",
                            "isBkg" : 1,
                            "xsec" : 0.60},
                     "WZ": {"path" : "WZ_TuneCP5_13TeV-pythia8/",
                            "isBkg" : 1,
                            "xsec" : 47.03}, # this value might be incorrect
                     "WW": {"path" : "WW_TuneCP5_13TeV-pythia8/",
                            "isBkg" : 1,
                            "xsec" : 118.7}, # this value might be incorrect
                    }

    allValidProcs = list(inputdir_dict.keys())
    allValidProcsAndSpecial = list(allValidProcs)
    allValidProcsAndSpecial.extend(["all", "bkg", "stand"]) # this can be removed once we run automatically on everything
    
    parser = common_parser()
    parser.add_argument('-i','--indir',  default=None, type=str, required=True,
                        help='Input directory with the root files inside (common path for data and MC)')
    parser.add_argument('-o','--outdir', default=None, type=str, required=True,
                        help='Output directory to store all root files')
    parser.add_argument('-d',  '--dryRun', action='store_true',
                        help='Do not execute commands, just print them')
    parser.add_argument('-r',  '--run', default="all", type=str, choices=allValidProcsAndSpecial,
                        help='Choose what to run, either data or MC, or both')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i','--indir', help='Input directory with the root files inside (common path for data and MC)',
                        type=str, default=None, required=True)

    parser.add_argument('-o','--outdir', help='Output directory to store all root files',
    					type=str, default=None, required=True)

    parser.add_argument('-d',  '--dryRun', action='store_true', help='Do not execute commands, just print them')

    parser.add_argument('-r',  '--run', help='Choose what to run, either data or MC, or both',
    					default=["all"], nargs="*",
    					choices=["data", "mc", "stand", "bkg", "Ztautau", "TTSemileptonic", "TTFullyleptonic", "WplusJets", "WminusJets", "ZZ", "WZ", "WW", "all"])
                    
    # FIXME: unless I change histogram names inside the files I can't merge different working points, I could just merge data with MC but not worth
    #parser.add_argument('-m',  '--merge', action='store_true',
    #                    help='Merge root files in a new one')
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
    #parser.add_argument('-exe', '--executable', default="Steve.py", type=str, choices=["Steve.py", "Steve_tracker.py"],
    #                    help='Choose script to run')
    #parser.add_argument('-m',  '--merge', action='store_true', help='Merge root files in a new one')
   
    parser.add_argument('-nw', '--noVertexPileupWeight', action='store_true', help='Do not use weights for vertex z position')
    
    parser.add_argument('-ngm', '--noGenMatching', action='store_true', help='Do not apply gen-matching for the probe (for non prompt bkg study)')
    
    parser.add_argument(        '--reverseGenMatching', action='store_true', help='Reverse the gen-matching condition for the probe (for non prompt contributions on Zmumu sample)')
    
    parser.add_argument("-nos", "--noOppositeCharge", action="store_true", help="Don't require opposite charges between tag and probe (including tracking, unless also using --noOppositeChargeTracking)")
    
    parser.add_argument(        "--noOppositeChargeTracking", action="store_true", help="Don't require opposite charges between tag and probe for tracking")
    
    parser.add_argument("-sc", "--SameCharge", action="store_true", help="Require the TP Pair to have same sign (fo bkg study)")
    
    parser.add_argument('-s', '--steps', help='Default runs all working points, but can choose only some if needed',
    					type=int, default=None, nargs='*', choices=list(workingPoints.keys()))

    parser.add_argument('-x','--exclude', help='Default runs all working points, but can choose to skip some if needed',
    					type=int, default=None, nargs='*', choices=list(workingPoints.keys()))
    					
    parser.add_argument('-wpc','---workinPointsByCharge', help='These steps will be made charge dependent',
    					type=str, default=["trigger"], nargs='*', choices=list(workingPoints.values()))
    					
    parser.add_argument("-trk", "--trackerMuons", action="store_true", help="Use tracker muons and a different executable")
    
    parser.add_argument("-p", "--eventParity", help="Select events with given parity for statistical tests, -1/1 for odd/even events, 0 for all (default)",
                        type=int, default=[0], nargs='+', choices=[-1, 0, 1])
              
    parser.add_argument("-tpt", "--tagPt", help="Minimum pt to select tag muons",
                        type=float, default=25.)
                        
    parser.add_argument("-tiso", "--tagIso", help="Isolation threshold to select tag muons",
                        type=float, default=0.15)
                        
    parser.add_argument(		"--standaloneValidHits", help="Minimum number of valid hits for the standalone track (>= this value)",
                        type=int, default=1)
                        
    # parser.add_argument('-exe', '--executable', help='Choose script to run',
    # 					  type=str, default="Steve.py", choices=["Steve.py", "Steve_tracker.py"])

    parser.add_argument("-y", "--year", help="run year 2016, 2017, 2018",
                    	type=str, default="2016")
                    	
    parser.add_argument("-iso", "--isoDefinition", help="Choose between the old and new isolation definition, 0 is old, 1 is new", 
    					type=int, default=1, choices = [0,1])


    args = parser.parse_args()

    # compare pt values within some tolerance
    if (args.histMinPt + 0.01) < args.innerTrackMinPt:
        raise IOError(f"Inconsistent values for options --histMinPt ({args.histMinPt}) and --innerTrackMinPt ({args.innerTrackMinPt}).\nThe former must not be smaller than the latter.\n")

    outdir = args.outdir
    if not outdir.endswith("/"): outdir += "/"
    
    indir = args.indir
    if not indir.endswith("/"):  indir += "/"


    executable = "Steve_tracker.py" if args.trackerMuons else "Steve.py"

    if not os.path.exists(outdir):
        print(f"Creating folder {outdir}")
        safeSystem(f"mkdir -p {outdir}", dryRun=False)

    
    toRun = []
    for p in allValidProcs:
        if args.run == "all" or args.run == p or (args.run == "bgk" and inputdir_dict[p]["isBkg"]) or (args.run == "stand" and p in ["data", "mc"]):
            toRun.append(p)

    for dataset_run in args.run:
        if dataset_run in ["all", "data","stand"]: toRun.append("data")
        if dataset_run in ["all", "mc",  "stand"]: toRun.append("mc")
        if dataset_run in ["all", "bkg", "Ztautau"]: 		  toRun.append("Ztautau")
        if dataset_run in ["all", "bkg", "TTFullyleptonic"]: toRun.append("TTFullyleptonic")
        if dataset_run in ["all", "bkg", "TTSemileptonic"]:  toRun.append("TTSemileptonic")
        if dataset_run in ["all", "bkg", "WplusJets"]:	 toRun.append("WplusJets")
        if dataset_run in ["all", "bkg", "WminusJets"]: toRun.append("WminusJets")
        if dataset_run in ["all", "bkg", "ZZ"]: toRun.append("ZZ")
        if dataset_run in ["all", "bkg", "WZ"]: toRun.append("WZ")
        if dataset_run in ["all", "bkg", "WW"]: toRun.append("WW")
   
   
    outfiles = [] # store names of output files so to merge them if needed

    postfix = "vertexWeights{v}".format(v="0" if args.noVertexPileupWeight else "1")
    if args.SameCharge:
        postfix += "_sscharge"
    else:
        postfix += "_oscharge{c}".format(c="0" if args.noOppositeCharge else "1")
    
    sign_vw = "0" if args.noVertexPileupWeight else "1"
    sign_gm = "0" if args.noGenMatching 	   else "1"
    if args.reverseGenMatching: 
    	sign_gm = "-1"
    sign_os = "0" if args.noOppositeCharge	   else "1"
    SS_opt = "_SS" if args.SameCharge 	       else ""
    
    postfix = f"vertexWeights{sign_vw}_genMatching{sign_gm}_oscharge{sign_os}{SS_opt}"
                                                    				              

    commonOption = f" --tagPt {args.tagPt} --tagIso {args.tagIso} --standaloneValidHits {args.standaloneValidHits}"

    for xrun in toRun:

        process = inputdir_dict[xrun]
        #inpath = indir + (inputdir_data if isdata else inputdir_mc)
        if isinstance(process["path"], list):
            inpath = " ".join(["{i}{f}".format(i=indir,f=x) for x in process["path"]])
        else:
            inpath = indir + process["path"]
        
        input_datasets = [inputdir_dict[xrun]] if type(inputdir_dict[xrun]) is str else inputdir_dict[xrun]
        inpaths = [indir + in_path for in_path in input_datasets]
        
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
                    if wp in [2, 9] and args.noOppositeChargeTracking and not args.SameCharge:
                        postfixTracking = postfix.replace("oscharge1", "oscharge0")
                        outfile = f"{outdir}tnp_{step}_{xrun}_{postfixTracking}.root"
                    else:
                        outfile = f"{outdir}tnp_{step}_{xrun}_{postfix}.root"                        
                    outfiles.append(outfile)
                    
                    inpaths_cmd = ""
                    for inpath in inpaths: 
                        inpaths_cmd = inpaths_cmd + inpath + " "
                    
                    
                    cmd = f"python {executable} -i {inpaths_cmd} -o {outfile} -d {isdata} -b {isBkg} -e {wp} -c {ch} -p {parity} -y {args.year} -iso {args.isoDefinition}"
                    cmd += commonOption
                    if args.noVertexPileupWeight:
                        cmd += " -nw"
                    if args.noGenMatching:
                    	cmd += " -ngm"
                    if args.reverseGenMatching:
                    	cmd += " --reverseGenMatching"
                    if args.noOppositeCharge:
                        cmd += " -nos"
                    if wp in [2, 9] and not args.noOppositeChargeTracking:
                        cmd += " --oppositeChargeTracking "
                    if args.noOppositeChargeTracking and wp == 2:
                        cmd += " --noOppositeChargeTracking"
                    if args.SameCharge:
                        cmd += " --SameCharge"
                    # pt customization options
                    cmd += f" --histMinPt {args.histMinPt} --innerTrackMinPt {args.innerTrackMinPt} --standaloneMinPt {args.standaloneMinPt} "
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
