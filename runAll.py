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

inputdir_dict = { "data" : "SingleMuon/", 
                  "mc" : ["DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/",
                          "DYJetsToMuMu_H2ErratumFix_PDFExt_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/"
                          #"DYJetsToMuMu_M-10to50_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/"
                          ],
                  "Ztautau" : "DYJetsToTauTau_M-50_AtLeastOneEorMuDecay_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/",
                  "TTFullyleptonic" : "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/",
                  "TTSemileptonic" : "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/",
                  "WplusJets" : "WplusJetsToMuNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/",
                  "WminusJets" : "WminusJetsToMuNu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/",
                  "ZZ" : "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/", 
                  "WZ" : "WZ_TuneCP5_13TeV-pythia8/", 
                  "WW" : "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/", #"WW_TuneCP5_13TeV-pythia8/"
                  "QCD" : "QCD_Pt-20_MuEnrichedPt15_TuneCP5_13TeV-pythia8/" 
}
#inputdir_data = "SingleMuon/"
#inputdir_mc   = "DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/"
#inputdir_Ztautau = "DYJetsToTauTau_M-50_AtLeastOneEorMuDecay_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/"
#inputdir_TTSemileptonic = "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/"
#inputdir_ZZ = "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/"
#inputdir_WZ = "WZ_TuneCP5_13TeV-pythia8/"
#inputdir_WW = "WW_TuneCP5_13TeV-pythia8/"   

isBkg_dict = {"data": 0, "mc": 0, "Ztautau": 1, "TTSemileptonic": 1, "TTFullyleptonic": 1, "WplusJets": 1, "WminusJets":1, "ZZ": 1, "WZ": 1, "WW": 1, "QCD": 1}


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i','--indir', help='Input directory with the root files inside (common path for data and MC)',
                        type=str, default=None, required=True)

    parser.add_argument('-o','--outdir', help='Output directory to store all root files',
    					type=str, default=None, required=True)

    parser.add_argument('-d',  '--dryRun', action='store_true', help='Do not execute commands, just print them')

    parser.add_argument('-r',  '--run', help='Choose what to run, either data or MC, or both',
    					default=["all"], nargs="*",
    					choices=["data", "mc", "stand", "bkg", "Ztautau", "TTSemileptonic", "TTFullyleptonic", "WplusJets", "WminusJets", "ZZ", "WZ", "WW", "QCD", "all"])
                    
    # FIXME: unless I change histogram names inside the files I can't merge different working points, I could just merge data with MC but not worth
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

    outdir = args.outdir
    if not outdir.endswith("/"): outdir += "/"
    
    indir = args.indir
    if not indir.endswith("/"):  indir += "/"


    executable = "Steve_tracker.py" if args.trackerMuons else "Steve.py"
        
    if not os.path.exists(outdir):
        print(f"Creating folder {outdir}")
        safeSystem(f"mkdir -p {outdir}", dryRun=False)
    
    toRun = []
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
        if dataset_run in ["all", "bkg", "QCD"]: toRun.append("QCD")
   
   
    outfiles = [] # store names of output files so to merge them if needed
    
    sign_vw = "0" if args.noVertexPileupWeight else "1"
    sign_gm = "0" if args.noGenMatching 	   else "1"
    if args.reverseGenMatching: 
    	sign_gm = "-1"
    sign_os = "0" if args.noOppositeCharge	   else "1"
    SS_opt = "_SS" if args.SameCharge 	       else ""
    
    postfix = f"vertexWeights{sign_vw}_genMatching{sign_gm}_oscharge{sign_os}{SS_opt}"
                                                    				              

    commonOption = f" --tagPt {args.tagPt} --tagIso {args.tagIso} --standaloneValidHits {args.standaloneValidHits}"

    for xrun in toRun:

        isdata = 1 if xrun == "data" else 0
        isBkg = isBkg_dict[xrun]
        #inpath = indir + (inputdir_data if isdata else inputdir_mc)
        
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
                    if wp == 2 and args.noOppositeChargeTracking:
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
                    if args.noOppositeChargeTracking and wp == 2:
                        cmd += " --noOppositeChargeTracking"
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
