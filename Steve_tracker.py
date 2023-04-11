import ROOT
from array import array
import json

ROOT.gInterpreter.ProcessLine(".O3")
ROOT.ROOT.EnableImplicitMT()
ROOT.gInterpreter.Declare('#include "Steve.h"')
ROOT.gInterpreter.Declare('#include "GenFunctions.h"')
import os
import time

import argparse


def makeAndSaveOneHist(d, histo_name, histo_title, binning_mass, binning_pt, binning_eta, massVar="TPmass", isPass=True):

    passStr = "pass" if isPass else "fail"
    model = ROOT.RDF.TH3DModel(f"{passStr}_mu_{histo_name}", f"{histo_title} {passStr}",
                               len(binning_mass)-1, binning_mass,
                               len(binning_pt)-1, binning_pt,
                               len(binning_eta)-1, binning_eta)
    
    histogram = d.Histo3D(model, f"{massVar}_{passStr}", f"Probe_pt_{passStr}", f"Probe_eta_{passStr}", "weight")
    histogram.Write()
    
    
def makeAndSaveHistograms(d, histo_name, histo_title, binning_mass, binning_pt, binning_eta, massVar="TPmass"):

    makeAndSaveOneHist(d, histo_name, histo_title, binning_mass, binning_pt, binning_eta, massVar, isPass=True)
    makeAndSaveOneHist(d, histo_name, histo_title, binning_mass, binning_pt, binning_eta, massVar, isPass=False)
    


def make_jsonhelper(filename):
    with open(filename) as jsonfile:
        jsondata = json.load(jsonfile)
    
    runs = []
    firstlumis = []
    lastlumis = []
    
    for run,lumipairs in jsondata.items():
        for lumipair in lumipairs:
            runs.append(int(run))
            firstlumis.append(int(lumipair[0]))
            lastlumis.append(int(lumipair[1]))
    
    jsonhelper = ROOT.JsonHelper(runs, firstlumis, lastlumis)
    
    return jsonhelper

parser = argparse.ArgumentParser()

parser.add_argument("-e","--efficiency",
		    help="1 for reco, 2 for \"tracking\", 3 for idip, 4 for trigger, 5 for isolation, 6 for isolation without trigger, 7 for veto",
                    type=int, choices=range(1,8))
parser.add_argument("-i","--input_path", help="path of the input root files",
                    type=str)

parser.add_argument("-o","--output_file", help="name of the output root file",
                    type=str)

parser.add_argument("-d","--isData", help="Pass 0 for MC, 1 for Data, default is 0",
                    type=int, default=0)

parser.add_argument("-tpt","--tagPt", help="Minimum pt to select tag muons",
                    type=float, default=25.)

parser.add_argument("-c","--charge", help="Make efficiencies for a specific charge of the probe (-1/1 for positive negative, 0 for inclusive)",
                    type=int, default=0, choices=[-1, 0, 1])

parser.add_argument('-nw', '--noVertexPileupWeight', action='store_true', help='Do not use weights for vertex z position')
#parser.add_argument("-vpw", "--vertexPileupWeight", action="store_true", help="Use weights for vertex z position versus pileup (only for MC)")

parser.add_argument("-nos", "--noOppositeCharge", action="store_true", help="Don't require opposite charges between tag and probe (including tracking, unless also using --noOppositeChargeTracking)")
parser.add_argument(        "--noOppositeChargeTracking", action="store_true", help="Don't require opposite charges between tag and probe for tracking")

parser.add_argument("-zqt","--zqtprojection", action="store_true", help="Efficiencies evaluated as a function of zqtprojection (only for trigger and isolation)")

parser.add_argument("-gen","--genLevelEfficiency", action="store_true", help="Compute MC truth efficiency")

parser.add_argument("-tpg","--tnpGenLevel", action="store_true", help="Compute tag-and-probe efficiencies for MC as a function of postVFP gen variables")

args = parser.parse_args()
tstart = time.time()
cpustrat = time.process_time()

if args.isData & args.genLevelEfficiency:
    raise RuntimeError('\'genLevelEfficiency\' option not supported for data')

if args.isData & args.tnpGenLevel:
    raise RuntimeError('\'tnpGenLevel\' option not supported for data')

if '.root' not in args.output_file:
    raise NameError('output_file name must end with \'.root\'')

# create output folders if not existing
outdir = os.path.dirname(os.path.abspath(args.output_file))
if not os.path.exists(outdir):
    print()
    print(f"Creating folder {outdir} to store outputs")
    os.makedirs(outdir)    
    print()

if args.isData == 1:
    histo_name= "RunGtoH"
else:
    histo_name = "DY_postVFP"

files=[]

for root, dirnames, filenames in os.walk(args.input_path):
     for filename in filenames:
          if '.root' in filename:
              files.append(os.path.join(root, filename))


if args.charge and args.efficiency in [2]:
    print("")
    print("   WARNING: charge splitting not implemented for tracking efficiency. I will derive charge inclusive efficiencies")
    print("")


#print(files)
filenames = ROOT.std.vector('string')()

for name in files: filenames.push_back(name)

d = ROOT.RDataFrame("Events",filenames )

binning_pt = array('d',[24., 26., 28., 30., 32., 34., 36., 38., 40., 42., 44., 47., 50., 55., 60., 65.])
binning_eta = array('d',[round(-2.4 + i*0.1,2) for i in range(49)])
binning_mass = array('d',[50 + i for i in range(81)])
binning_charge = array('d',[-1.5,0,1.5])
binning_u = array('d',[-100 + i*2 for i in range(101)])

NBIN = ROOT.std.vector('int')()
NBIN.push_back(len(binning_mass)-1)
NBIN.push_back(len(binning_pt)-1)
NBIN.push_back(len(binning_eta)-1)
NBIN.push_back(len(binning_charge)-1)
NBIN.push_back(len(binning_u)-1)
XBINS = ROOT.std.vector('vector<double>')()
XBINS.push_back(ROOT.std.vector('double')(binning_mass))
XBINS.push_back(ROOT.std.vector('double')(binning_pt))
XBINS.push_back(ROOT.std.vector('double')(binning_eta))
XBINS.push_back(ROOT.std.vector('double')(binning_charge))
XBINS.push_back(ROOT.std.vector('double')(binning_u))
GENNBIN = ROOT.std.vector('int')()
GENNBIN.push_back(len(binning_pt)-1)
GENNBIN.push_back(len(binning_eta)-1)
GENNBIN.push_back(len(binning_charge)-1)
GENNBIN.push_back(len(binning_u)-1)
GENXBINS = ROOT.std.vector('vector<double>')()
GENXBINS.push_back(ROOT.std.vector('double')(binning_pt))
GENXBINS.push_back(ROOT.std.vector('double')(binning_eta))
GENXBINS.push_back(ROOT.std.vector('double')(binning_charge))
GENXBINS.push_back(ROOT.std.vector('double')(binning_u))

##General Cuts
d = d.Filter("HLT_IsoMu24 || HLT_IsoTkMu24","HLT Cut")

d = d.Filter("PV_npvsGood >= 1","NVtx Cut")

doOS = 0 if args.noOppositeCharge else 1
doOStracking = 0 if args.noOppositeChargeTracking else doOS

if (args.isData == 1):
    jsonhelper = make_jsonhelper("Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")
    d = d.Filter(jsonhelper,["run","luminosityBlock"],"jsonfilter")

## Weights

if(args.isData == 1):
    d = d.Define("weight","1")
else:
    if not args.noVertexPileupWeight:
        if hasattr(ROOT, "initializeVertexPileupWeights"):
            print("Initializing histograms with vertex-pileup weights")
            ROOT.initializeVertexPileupWeights("./utility/vertexPileupWeights.root")
            d = d.Define("vertex_weight", "_get_vertexPileupWeight(GenVtx_z,Pileup_nTrueInt,2)")
    else:
        d = d.Define("vertex_weight", "1.0")
    d = d.Define("gen_weight", "clipGenWeight(genWeight)") #for now (testing)
    d = d.Define("pu_weight", "puw_2016(Pileup_nTrueInt,2)")
    d = d.Define("weight", "gen_weight*pu_weight*vertex_weight")
    
## For Tag Muons
d = d.Define("isTriggeredMuon","hasTriggerMatch(Muon_eta, Muon_phi, TrigObj_id, TrigObj_filterBits, TrigObj_eta, TrigObj_phi)")

if(args.isData == 1):
    d = d.Define("isGenMatchedMuon","createTrues(nMuon)")
else: 
    d = d.Define("GenMuonBare", "GenPart_status == 1 && (GenPart_statusFlags & 1) && abs(GenPart_pdgId) == 13")
    d = d.Define("GenMuonBare_pt", "GenPart_pt[GenMuonBare]")
    d = d.Define("GenMuonBare_eta", "GenPart_eta[GenMuonBare]")
    d = d.Define("GenMuonBare_phi", "GenPart_phi[GenMuonBare]")
    d = d.Define("GenMuonBare_pdgId", "GenPart_pdgId[GenMuonBare]")
    d = d.Define("isGenMatchedMuon", "hasGenMatch(GenMuonBare_eta, GenMuonBare_phi, Muon_eta, Muon_phi)")

## Define tags as trigger matched and gen matched (gen match can be removed with an option in case)
d = d.Define("Tag_Muons", f"Muon_pt > {args.tagPt} && abs(Muon_eta) < 2.4 && Muon_pfRelIso04_all < 0.15 && abs(Muon_dxybs) < 0.05 && Muon_mediumId && isTriggeredMuon && isGenMatchedMuon")
# just for utility
d = d.Alias("Tag_pt",  "Muon_pt")
d = d.Alias("Tag_eta", "Muon_eta")
d = d.Alias("Tag_phi", "Muon_phi")
d = d.Alias("Tag_charge", "Muon_charge")
d = d.Alias("Tag_inExtraIdx", "Muon_innerTrackExtraIdx")
d = d.Alias("Tag_outExtraIdx", "Muon_standaloneExtraIdx")

if (args.genLevelEfficiency):
    d = d.Define("zero","0").Define("one","1") # is this really needed? Can't we just pass 1 or 0 in the functions where we need it?
    d = d.Define("GenPart_postFSRLepIdx1","PostFSRIdx(GenPart_pdgId,GenPart_status,GenPart_genPartIdxMother,GenPart_statusFlags,GenPart_pt,zero)")
    d = d.Define("GenPart_postFSRLepIdx2","PostFSRIdx(GenPart_pdgId,GenPart_status,GenPart_genPartIdxMother,GenPart_statusFlags,GenPart_pt,one)")
    d = d.Define("goodgenpt","goodgenvalue(GenPart_pt,GenPart_postFSRLepIdx1,GenPart_postFSRLepIdx2,GenPart_eta,GenPart_phi,GenPart_status,GenPart_pdgId)")
    d = d.Define("goodgeneta","goodgenvalue(GenPart_eta,GenPart_postFSRLepIdx1,GenPart_postFSRLepIdx2,GenPart_eta,GenPart_phi,GenPart_status,GenPart_pdgId)")
    d = d.Define("goodgenphi","goodgenvalue(GenPart_phi,GenPart_postFSRLepIdx1,GenPart_postFSRLepIdx2,GenPart_eta,GenPart_phi,GenPart_status,GenPart_pdgId)")
    d = d.Define("goodgencharge","goodgencharge(GenPart_postFSRLepIdx1,GenPart_postFSRLepIdx2,GenPart_eta,GenPart_phi,GenPart_status,GenPart_pdgId)")
    d = d.Define("goodgenidx","goodgenidx(GenPart_pt,GenPart_postFSRLepIdx1,GenPart_postFSRLepIdx2,GenPart_eta,GenPart_phi,GenPart_status,GenPart_pdgId)")
    d = d.Define("postFSRgenzqtprojection","postFSRgenzqtprojection(goodgenpt,goodgeneta,goodgenphi)")
    # this might be done simply as
    # d = d.Define("goodgenpt", "GenMuonBare_pt") # the collection might have more than 2 elements here, but can be easily filtered (should be sorted too)

# Open output file
f_out = ROOT.TFile(args.output_file,"RECREATE")
    
## Tracks for reco efficiency
if(args.efficiency == 1):
    if not (args.genLevelEfficiency):

        if(args.isData == 1):
            d = d.Define("isGenMatchedTrack","createTrues(nTrack)")
        else:
            d = d.Define("isGenMatchedTrack", "hasGenMatch(  GenMuonBare_eta, GenMuonBare_phi, Track_eta, Track_phi)")
            d = d.Define("GenMatchedIdx",     "GenMatchedIdx(GenMuonBare_eta, GenMuonBare_phi, Track_eta, Track_phi)")

        chargeCut = ""
        if args.charge:
            sign= ">" if args.charge > 0 else "<"
            chargeCut = f" && Track_charge {sign} 0"
            
        # define all probes
        d = d.Define("Probe_Tracks", f"Track_pt > 24 && abs(Track_eta) < 2.4 && Track_trackOriginalAlgo != 13 && Track_trackOriginalAlgo != 14 && isGenMatchedTrack && (Track_qualityMask & 4) {chargeCut}")
        #d = d.Filter("Sum(Probe_Tracks) > 0")
        
        d = d.Define("All_TPPairs", f"CreateTPPair(Tag_Muons, Probe_Tracks, {doOS}, Tag_charge, Track_charge, Tag_inExtraIdx, Track_extraIdx)")
        d = d.Define("All_TPmass", "getTPmass(All_TPPairs, Tag_pt, Tag_eta, Tag_phi, Track_pt, Track_eta, Track_phi)")
        
        # overriding previous pt binning
        binning_pt = array('d',[24., 26., 30., 34., 38., 42., 46., 50., 55., 65.])
        massLow  =  60
        massHigh = 120
        binning_mass = array('d',[massLow + i for i in range(int(1+massHigh-massLow))])
        massCut = f"All_TPmass > {massLow} && All_TPmass < {massHigh}"
        d = d.Define("TPPairs", f"All_TPPairs[{massCut}]")
        d = d.Define("TPmass",  f"All_TPmass[{massCut}]")
        
        d = d.Define("Probe_pt",   "getVariables(TPPairs, Track_pt,  2)")
        d = d.Define("Probe_eta",  "getVariables(TPPairs, Track_eta, 2)")
        if (args.tnpGenLevel):
            d = d.Redefine("Probe_pt","getGenVariables(TPPairs,GenMatchedIdx,GenMuonBare_pt,2)")
            d = d.Redefine("Probe_eta","getGenVariables(TPPairs,GenMatchedIdx,GenMuonBare_eta,2)")

        # condition for passing probes
        d = d.Define("trackerMuons", "Muon_pt > 10 && Muon_isTracker && Muon_innerTrackOriginalAlgo != 13 && Muon_innerTrackOriginalAlgo != 14 && Muon_highPurity")
        # check Muon exists with proper criteria and matching extraIdx with the track 

        # # TEST 1 (it runs fine with coll1coll2DR instead of Probe_isMatched, but I don't know if the result would be the same)
        # d = d.Define("passCondition_reco",
        #              #"coll1coll2DR(Track_eta, Track_phi, Muon_eta[trackerMuons], Muon_phi[trackerMuons]) < 0.1")
        #              "Probe_isMatched(TPPairs, Track_extraIdx, Muon_innerTrackExtraIdx, trackerMuons)")
        # d = d.Define("passCondition", "getVariables(TPPairs, passCondition_reco, 2)")
        # d = d.Define("failCondition", "!passCondition")
        
        d = d.Define("passCondition_reco",
                     #"coll1coll2DR(Track_eta, Track_phi, Muon_eta[trackerMuons], Muon_phi[trackerMuons]) < 0.1")
                     "Probe_isMatched(TPPairs, Track_extraIdx, Muon_innerTrackExtraIdx, trackerMuons)")
        d = d.Define("passCondition", "getVariables(TPPairs, passCondition_reco, 2)")
        d = d.Define("failCondition", "!passCondition")

        
        d = d.Define("Probe_pt_pass",  "Probe_pt[passCondition]")
        d = d.Define("Probe_eta_pass", "Probe_eta[passCondition]")
        d = d.Define("TPmass_pass", "TPmass[passCondition]")
        d = d.Define("Probe_pt_fail",  "Probe_pt[failCondition]")
        d = d.Define("Probe_eta_fail", "Probe_eta[failCondition]")
        d = d.Define("TPmass_fail", "TPmass[failCondition]")
        makeAndSaveHistograms(d, histo_name, "Reco", binning_mass, binning_pt, binning_eta)
        
    else:
        d = d.Define("goodmuon","goodmuonreco(goodgeneta,goodgenphi,MergedStandAloneMuon_pt,MergedStandAloneMuon_eta,MergedStandAloneMuon_phi)").Define("newweight","weight*goodmuon")

        model_pass_reco = ROOT.RDF.TH2DModel("Pass","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)
        model_norm_reco = ROOT.RDF.TH2DModel("Norm","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)

        pass_histogram_reco = d.Histo2D(model_pass_reco,"goodgeneta","goodgenpt","newweight")
        pass_histogram_norm = d.Histo2D(model_norm_reco,"goodgeneta","goodgenpt","weight")

        pass_histogram_reco.Write()
        pass_histogram_norm.Write()


#Global|MergedStandAloneMuon ("tracking" efficiency)
elif (args.efficiency == 2):
    if not (args.genLevelEfficiency):
        if(args.isData == 1):
            d = d.Define("isGenMatchedMergedStandMuon","createTrues(nMergedStandAloneMuon)")
        else:
            d = d.Define("isGenMatchedMergedStandMuon","hasGenMatch(GenMuonBare_eta, GenMuonBare_phi, MergedStandAloneMuon_eta, MergedStandAloneMuon_phi, 0.3)")
            d = d.Define("GenMatchedIdx","GenMatchedIdx(GenMuonBare_eta, GenMuonBare_phi, MergedStandAloneMuon_eta, MergedStandAloneMuon_phi, 0.3)")

        # All probes, standalone muons from MergedStandAloneMuon_XX    
        d = d.Define("Probe_MergedStandMuons","MergedStandAloneMuon_pt > 15 && abs(MergedStandAloneMuon_eta) < 2.4 && isGenMatchedMergedStandMuon")

        # note: the Tag_outExtraIdx might be filled with junk if the muon is not global, but for the purpose of disentangling with SA muon it is fine, they will never match
        d = d.Define("All_TPPairs", f"CreateTPPair(Tag_Muons, Probe_MergedStandMuons, {doOStracking}, Tag_charge, MergedStandAloneMuon_charge, Tag_outExtraIdx, MergedStandAloneMuon_extraIdx)")
        d = d.Define("All_TPmass","getTPmass(All_TPPairs, Tag_pt, Tag_eta, Tag_phi, MergedStandAloneMuon_pt, MergedStandAloneMuon_eta, MergedStandAloneMuon_phi)")

        massLow  =  50
        massHigh = 130
        binning_mass = array('d',[massLow + i for i in range(int(1+massHigh-massLow))])
        massCut = f"All_TPmass > {massLow} && All_TPmass < {massHigh}"
        d = d.Define("TPPairs", f"All_TPPairs[{massCut}]")
        d = d.Define("TPmass",  f"All_TPmass[{massCut}]")

        d = d.Define("Probe_pt",  "getVariables(TPPairs, MergedStandAloneMuon_pt,  2)")
        d = d.Define("Probe_eta", "getVariables(TPPairs, MergedStandAloneMuon_eta, 2)")

        if (args.tnpGenLevel):
            d = d.Redefine("Probe_pt","getGenVariables(TPPairs,GenMatchedIdx,GenMuonBare_pt,2)")
            d = d.Redefine("Probe_eta","getGenVariables(TPPairs,GenMatchedIdx,GenMuonBare_eta,2)")

        coneDR = 0.3
        # get good tracks
        d = d.Define("Track_forTracking", f"Track_pt > 15 && Track_trackOriginalAlgo != 13 && Track_trackOriginalAlgo != 14 && (Track_qualityMask & 4)")
        # get mask for SA muons which are matched to good tracks within DR
        d = d.Define("standaloneTrackDR",
                     f"coll1coll2DR(MergedStandAloneMuon_eta, MergedStandAloneMuon_phi, Track_eta[Track_forTracking], Track_phi[Track_forTracking]) < {coneDR}")
        d = d.Define("passCondition_tracking", "getVariables(TPPairs, standaloneTrackDR,  2)") 
                     
        binning_pt = array('d',[25., 35., 45., 55., 65.])

        d = d.Define("TPPairs_pass", "TPPairs[passCondition_tracking]")
        # now get the idx of the track that is matched to the standalone muon, to fill the passing mass later
        # if more tracks are found, we currently take the one with largest pt
        # the DR selection is repeated here
        d = d.Define("MergedStandaloneMuon_trackIdx",
                     f"getMergedStandAloneMuon_highestPtTrackIdxWithinDR(MergedStandAloneMuon_eta, MergedStandAloneMuon_phi, Track_pt[Track_forTracking], Track_eta[Track_forTracking], Track_phi[Track_forTracking], {coneDR})")
        d = d.Define("MergedStandaloneMuon_trackPt",  "getMergedStandAloneMuon_matchedObjectVar(MergedStandaloneMuon_trackIdx, Track_pt)")
        d = d.Define("MergedStandaloneMuon_trackEta", "getMergedStandAloneMuon_matchedObjectVar(MergedStandaloneMuon_trackIdx, Track_eta)")
        d = d.Define("MergedStandaloneMuon_trackPhi", "getMergedStandAloneMuon_matchedObjectVar(MergedStandaloneMuon_trackIdx, Track_phi)")

        # mass filled with track variables here
        d = d.Define("TPmass_pass",    "getTPmass(TPPairs_pass, Tag_pt, Tag_eta, Tag_phi, MergedStandaloneMuon_trackPt, MergedStandaloneMuon_trackEta, MergedStandaloneMuon_trackPhi)")
        d = d.Define("Probe_pt_pass",  "Probe_pt[passCondition_tracking]")
        d = d.Define("Probe_eta_pass", "Probe_eta[passCondition_tracking]")

        d = d.Define("TPmass_fail",    "TPmass[!passCondition_tracking]")
        d = d.Define("Probe_pt_fail",  "Probe_pt[!passCondition_tracking]")
        d = d.Define("Probe_eta_fail", "Probe_eta[!passCondition_tracking]")

        makeAndSaveHistograms(d, histo_name, "Tracking", binning_mass, binning_pt, binning_eta)

        # save also the mass for passing probes computed with standalone variables
        # needed when making MC template for failing probes using all probes, since the mass should be consistently measured for both cases
        # do it also for data in case we want to check the difference in the efficiencies
        d = d.Define("TPmassFromSA_pass", "TPmass[passCondition_tracking]")
        makeAndSaveOneHist(d, f"{histo_name}_alt", "Tracking (mass from SA muons)",
                           binning_mass, binning_pt, binning_eta,
                           massVar="TPmassFromSA", isPass=True)
        
    else:
        d = d.Define("goodmuon","goodmuonglobal(goodgeneta,goodgenphi,Muon_pt,Muon_eta,Muon_phi,Muon_isGlobal,Muon_standalonePt,Muon_standaloneEta,Muon_standalonePhi)").Define("newweight","weight*goodmuon")

        model_pass_reco = ROOT.RDF.TH2DModel("Pass","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)
        model_norm_reco = ROOT.RDF.TH2DModel("Norm","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)

        pass_histogram_reco = d.Histo2D(model_pass_reco,"goodgeneta","goodgenpt","newweight")
        pass_histogram_norm = d.Histo2D(model_norm_reco,"goodgeneta","goodgenpt","weight")

        pass_histogram_reco.Write()
        pass_histogram_norm.Write()

## Muons for all other efficiency step except veto
elif args.efficiency != 7:
    if(args.isData != 1):
        d = d.Define("GenMatchedIdx","GenMatchedIdx(GenMuonBare_eta, GenMuonBare_phi, Muon_eta, Muon_phi)")

    chargeCut = ""
    if args.charge:
        sign= ">" if args.charge > 0 else "<"
        chargeCut = f" && Muon_charge {sign} 0"
        
    d = d.Define("BasicProbe_Muons", f"Muon_isTracker && Muon_pt > 24 && abs(Muon_eta) < 2.4 && Muon_innerTrackOriginalAlgo != 13 && Muon_innerTrackOriginalAlgo != 14 && Muon_highPurity && isGenMatchedMuon {chargeCut}")

    d = d.Define("All_TPPairs", f"CreateTPPair(Tag_Muons, BasicProbe_Muons, {doOS}, Tag_charge, Muon_charge, Tag_inExtraIdx, Muon_innerTrackExtraIdx, 1)") # these are all Muon_XX, so might just exclude same index in the loop
    d = d.Define("All_TPmass","getTPmass(All_TPPairs, Tag_pt, Tag_eta, Tag_phi, Muon_pt, Muon_eta, Muon_phi)")
    massLow  =  60
    massHigh = 120
    binning_mass = array('d',[massLow + i for i in range(int(1+massHigh-massLow))])
    massCut = f"All_TPmass > {massLow} && All_TPmass < {massHigh}"

    d = d.Define("TPPairs", f"All_TPPairs[{massCut}]")
    # call it BasicTPmass so it can be filtered later without using Redefine, but an appropriate Define
    d = d.Define("BasicTPmass",  f"All_TPmass[{massCut}]")

    ####
    ####
    # define all basic probes here (these are all Muon), to be filtered further later, without having to use Redefine when filtering
    d = d.Define("BasicProbe_charge", "getVariables(TPPairs, Muon_charge, 2)")
    d = d.Define("BasicProbe_pt",     "getVariables(TPPairs, Muon_pt,     2)")
    d = d.Define("BasicProbe_eta",    "getVariables(TPPairs, Muon_eta,    2)")
    d = d.Define("BasicProbe_u","zqtprojection(TPPairs,Muon_pt,Muon_eta,Muon_phi)")

    if (args.tnpGenLevel):
        d = d.Redefine("BasicProbe_pt",  "getGenVariables(TPPairs,GenMatchedIdx,GenMuonBare_pt,2)")
        d = d.Redefine("BasicProbe_eta", "getGenVariables(TPPairs,GenMatchedIdx,GenMuonBare_eta,2)")
        d = d.Redefine("BasicProbe_u",   "zqtprojectionGen(TPPairs,GenMatchedIdx,GenMuonBare_pt,GenMuonBare_eta,GenMuonBare_phi)")

    ## IMPORTANT: define only the specific condition to be passed, not with the && of previous steps (although in principle it is the same as long as that one is already applied)
    ##            also, these are based on the initial Muon collection, with no precooked filtering
    d = d.Define("passCondition_IDIP", "Muon_mediumId && abs(Muon_dxybs) < 0.05")
    d = d.Define("passCondition_Trig", "isTriggeredMuon")
    d = d.Define("passCondition_Iso",  "Muon_pfRelIso04_all < 0.15")
    #d = d.Define("passCondition_Iso",  "Muon_pfRelIso03_all < 0.10")
    #d = d.Define("passCondition_Iso",  "Muon_pfRelIso03_chg < 0.05")
    
    # For IDIP
    if (args.efficiency == 3):
        if not (args.genLevelEfficiency):
            # define condition for passing probes
            d = d.Define("passCondition", "getVariables(TPPairs, passCondition_IDIP, 2)")
            d = d.Define("failCondition", "!passCondition")            
            # pass probes
            d = d.Define("Probe_pt_pass",  "BasicProbe_pt[passCondition]")
            d = d.Define("Probe_eta_pass", "BasicProbe_eta[passCondition]")
            d = d.Define("TPmass_pass",    "BasicTPmass[passCondition]")
            # fail probes
            d = d.Define("Probe_pt_fail",  "BasicProbe_pt[failCondition]")
            d = d.Define("Probe_eta_fail", "BasicProbe_eta[failCondition]")
            d = d.Define("TPmass_fail",    "BasicTPmass[failCondition]")
            makeAndSaveHistograms(d, histo_name, "IDIP", binning_mass, binning_pt, binning_eta)
        else:
            d = d.Define("goodmuon","goodmuonglobal(goodgeneta,goodgenphi,Muon_pt,Muon_eta,Muon_phi,Muon_isGlobal,Muon_standalonePt,Muon_standaloneEta,Muon_standalonePhi,Muon_dxybs,Muon_mediumId)")
            d = d.Define("newweight","weight*goodmuon")

            model_pass_reco = ROOT.RDF.TH2DModel("Pass","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)
            model_norm_reco = ROOT.RDF.TH2DModel("Norm","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)

            pass_histogram_reco = d.Histo2D(model_pass_reco,"goodgeneta","goodgenpt","newweight")
            pass_histogram_norm = d.Histo2D(model_norm_reco,"goodgeneta","goodgenpt","weight")

            pass_histogram_reco.Write()
            pass_histogram_norm.Write()

    # For Trigger
    if(args.efficiency == 4):

        # define condition for passing probes
        d = d.Define("passCondition_IDIPTrig", "passCondition_IDIP &&  passCondition_Trig")
        d = d.Define("failCondition_IDIPTrig", "passCondition_IDIP && !passCondition_Trig")
        d = d.Define("passCondition", "getVariables(TPPairs, passCondition_IDIPTrig, 2)")
        d = d.Define("failCondition", "getVariables(TPPairs, failCondition_IDIPTrig, 2)")            
        # pass probes
        d = d.Define("Probe_pt_pass",  "BasicProbe_pt[passCondition]")
        d = d.Define("Probe_eta_pass", "BasicProbe_eta[passCondition]")
        d = d.Define("TPmass_pass",    "BasicTPmass[passCondition]")
        d = d.Define("Probe_u_pass",        "BasicProbe_u[passCondition]")
        d = d.Define("Probe_charge_pass",   "BasicProbe_charge[passCondition]")
        # fail probes
        d = d.Define("Probe_pt_fail",  "BasicProbe_pt[failCondition]")
        d = d.Define("Probe_eta_fail", "BasicProbe_eta[failCondition]")
        d = d.Define("TPmass_fail",    "BasicTPmass[failCondition]")        
        d = d.Define("Probe_u_fail",        "BasicProbe_u[failCondition]")
        d = d.Define("Probe_charge_fail",   "BasicProbe_charge[failCondition]")

        if (args.zqtprojection):
            if not (args.genLevelEfficiency):
                model_pass_trig = ROOT.RDF.THnDModel("pass_mu_"+histo_name, "Trigger_pass", 5, NBIN, XBINS)
                model_fail_trig = ROOT.RDF.THnDModel("fail_mu_"+histo_name, "Trigger_fail", 5, NBIN, XBINS)
                strings_pass = ROOT.std.vector('string')()
                strings_pass.emplace_back("TPmass_pass")
                strings_pass.emplace_back("Probe_pt_pass")
                strings_pass.emplace_back("Probe_eta_pass")
                strings_pass.emplace_back("Probe_charge_pass")
                strings_pass.emplace_back("Probe_u_pass")
                strings_pass.emplace_back("weight")
                strings_fail = ROOT.std.vector('string')()
                strings_fail.emplace_back("TPmass_fail")
                strings_fail.emplace_back("Probe_pt_fail")
                strings_fail.emplace_back("Probe_eta_fail")
                strings_fail.emplace_back("Probe_charge_fail")
                strings_fail.emplace_back("Probe_u_fail")
                strings_fail.emplace_back("weight")

                pass_histogram_trig = d.HistoND(model_pass_trig,strings_pass)

                fail_histogram_trig = d.HistoND(model_fail_trig,strings_fail)

                ROOT.saveHistograms(pass_histogram_trig,fail_histogram_trig,ROOT.std.string(args.output_file))
            else:
                model_pass_trig = ROOT.RDF.THnDModel("pass_mu_"+histo_name, "Trigger_pass", 4, GENNBIN, GENXBINS)
                model_norm_trig = ROOT.RDF.THnDModel("norm_mu_"+histo_name, "Trigger_norm", 4, GENNBIN, GENXBINS)
                strings_pass = ROOT.std.vector('string')()
                strings_pass.emplace_back("goodgenpt")
                strings_pass.emplace_back("goodgeneta")
                strings_pass.emplace_back("goodgencharge")
                strings_pass.emplace_back("postFSRgenzqtprojection")
                strings_pass.emplace_back("newweight")
                strings_norm = ROOT.std.vector('string')()
                strings_norm.emplace_back("goodgenpt")
                strings_norm.emplace_back("goodgeneta")
                strings_norm.emplace_back("goodgencharge")
                strings_norm.emplace_back("postFSRgenzqtprojection")
                strings_norm.emplace_back("weight")

                d = d.Define("goodmuon","goodmuontrigger(goodgeneta,goodgenphi,Muon_pt,Muon_eta,Muon_phi,Muon_isGlobal,Muon_standalonePt,Muon_standaloneEta,Muon_standalonePhi,Muon_dxybs,Muon_mediumId,isTriggeredMuon)").Define("newweight","weight*goodmuon")

                pass_histogram_reco = d.Filter("goodgenpt.size()>=2").HistoND(model_pass_trig,strings_pass)
                pass_histogram_norm = d.Filter("goodgenpt.size()>=2").HistoND(model_norm_trig,strings_norm)

                ROOT.saveHistogramsGen(pass_histogram_reco,pass_histogram_norm,ROOT.std.string(args.output_file))
                

        else:
            if not (args.genLevelEfficiency):
                makeAndSaveHistograms(d, histo_name, "Trigger", binning_mass, binning_pt, binning_eta)
            else:
                d = d.Define("goodmuon","goodmuontrigger(goodgeneta,goodgenphi,Muon_pt,Muon_eta,Muon_phi,Muon_isGlobal,Muon_standalonePt,Muon_standaloneEta,Muon_standalonePhi,Muon_dxybs,Muon_mediumId,isTriggeredMuon)").Define("newweight","weight*goodmuon")

                model_pass_reco = ROOT.RDF.TH2DModel("Pass","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)
                model_norm_reco = ROOT.RDF.TH2DModel("Norm","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)

                pass_histogram_reco = d.Histo2D(model_pass_reco,"goodgeneta","goodgenpt","newweight")
                pass_histogram_norm = d.Histo2D(model_norm_reco,"goodgeneta","goodgenpt","weight")

                pass_histogram_reco.Write()
                pass_histogram_norm.Write()

     ##For Isolation

    if(args.efficiency == 5):

        # define condition for passing probes
        d = d.Define("passCondition_IDIPTrigIso", "passCondition_IDIP && passCondition_Trig &&  passCondition_Iso")
        d = d.Define("failCondition_IDIPTrigIso", "passCondition_IDIP && passCondition_Trig && !passCondition_Iso")
        d = d.Define("passCondition", "getVariables(TPPairs, passCondition_IDIPTrigIso, 2)")
        d = d.Define("failCondition", "getVariables(TPPairs, failCondition_IDIPTrigIso, 2)")            
        # pass probes
        d = d.Define("Probe_pt_pass",  "BasicProbe_pt[passCondition]")
        d = d.Define("Probe_eta_pass", "BasicProbe_eta[passCondition]")
        d = d.Define("TPmass_pass",    "BasicTPmass[passCondition]")
        d = d.Define("Probe_u_pass",        "BasicProbe_u[passCondition]")
        d = d.Define("Probe_charge_pass",   "BasicProbe_charge[passCondition]")
        # fail probes
        d = d.Define("Probe_pt_fail",  "BasicProbe_pt[failCondition]")
        d = d.Define("Probe_eta_fail", "BasicProbe_eta[failCondition]")
        d = d.Define("TPmass_fail",    "BasicTPmass[failCondition]")        
        d = d.Define("Probe_u_fail",        "BasicProbe_u[failCondition]")
        d = d.Define("Probe_charge_fail",   "BasicProbe_charge[failCondition]")

        if (args.zqtprojection):
            if not (args.genLevelEfficiency):
                model_pass_iso = ROOT.RDF.THnDModel("pass_mu_"+histo_name, "Isolation_pass", 5, NBIN, XBINS)
                model_fail_iso = ROOT.RDF.THnDModel("fail_mu_"+histo_name, "Isolation_fail", 5, NBIN, XBINS)
                strings_pass = ROOT.std.vector('string')()
                strings_pass.emplace_back("TPmass_pass")
                strings_pass.emplace_back("Probe_pt_pass")
                strings_pass.emplace_back("Probe_eta_pass")
                strings_pass.emplace_back("Probe_charge_pass")
                strings_pass.emplace_back("Probe_u_pass")
                strings_pass.emplace_back("weight")
                strings_fail = ROOT.std.vector('string')()
                strings_fail.emplace_back("TPmass_fail")
                strings_fail.emplace_back("Probe_pt_fail")
                strings_fail.emplace_back("Probe_eta_fail")
                strings_fail.emplace_back("Probe_charge_fail")
                strings_fail.emplace_back("Probe_u_fail")
                strings_fail.emplace_back("weight")
     
                pass_histogram_iso = d.HistoND(model_pass_iso,strings_pass)
                fail_histogram_iso = d.HistoND(model_fail_iso,strings_fail)

                ROOT.saveHistograms(pass_histogram_iso,fail_histogram_iso,ROOT.std.string(args.output_file))
            else:
                model_pass_trig = ROOT.RDF.THnDModel("pass_mu_"+histo_name, "Isolation_pass", 4, GENNBIN, GENXBINS)
                model_norm_trig = ROOT.RDF.THnDModel("norm_mu_"+histo_name, "Isolation_norm", 4, GENNBIN, GENXBINS)
                strings_pass = ROOT.std.vector('string')()
                strings_pass.emplace_back("goodgenpt")
                strings_pass.emplace_back("goodgeneta")
                strings_pass.emplace_back("goodgencharge")
                strings_pass.emplace_back("postFSRgenzqtprojection")
                strings_pass.emplace_back("newweight")
                strings_norm = ROOT.std.vector('string')()
                strings_norm.emplace_back("goodgenpt")
                strings_norm.emplace_back("goodgeneta")
                strings_norm.emplace_back("goodgencharge")
                strings_norm.emplace_back("postFSRgenzqtprojection")
                strings_norm.emplace_back("weight")

                d = d.Define("goodmuon","goodmuonisolation(goodgeneta,goodgenphi,Muon_pt,Muon_eta,Muon_phi,Muon_isGlobal,Muon_standalonePt,Muon_standaloneEta,Muon_standalonePhi,Muon_dxybs,Muon_mediumId,isTriggeredMuon,Muon_pfRelIso04_all)").Define("newweight","weight*goodmuon")

                pass_histogram_reco = d.Filter("goodgenpt.size()>=2").HistoND(model_pass_trig,strings_pass)
                pass_histogram_norm = d.Filter("goodgenpt.size()>=2").HistoND(model_norm_trig,strings_norm)

                ROOT.saveHistogramsGen(pass_histogram_reco,pass_histogram_norm,ROOT.std.string(args.output_file))

        else:
            if not (args.genLevelEfficiency):
                makeAndSaveHistograms(d, histo_name, "Isolation", binning_mass, binning_pt, binning_eta)
            else:
                d = d.Define("goodmuon","goodmuonisolation(goodgeneta,goodgenphi,Muon_pt,Muon_eta,Muon_phi,Muon_isGlobal,Muon_standalonePt,Muon_standaloneEta,Muon_standalonePhi,Muon_dxybs,Muon_mediumId,isTriggeredMuon,Muon_pfRelIso04_all)").Define("newweight","weight*goodmuon")

                model_pass_reco = ROOT.RDF.TH2DModel("Pass","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)
                model_norm_reco = ROOT.RDF.TH2DModel("Norm","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)

                pass_histogram_reco = d.Histo2D(model_pass_reco,"goodgeneta","goodgenpt","newweight")
                pass_histogram_norm = d.Histo2D(model_norm_reco,"goodgeneta","goodgenpt","weight")

                pass_histogram_reco.Write()
                pass_histogram_norm.Write()

    # isolation without trigger
    if(args.efficiency == 6):

        # define condition for passing probes
        d = d.Define("passCondition_IDIPIso", "passCondition_IDIP &&  passCondition_Iso")
        d = d.Define("failCondition_IDIPIso", "passCondition_IDIP && !passCondition_Iso")
        d = d.Define("passCondition", "getVariables(TPPairs, passCondition_IDIPIso, 2)")
        d = d.Define("failCondition", "getVariables(TPPairs, failCondition_IDIPIso, 2)")            
        # pass probes
        d = d.Define("Probe_pt_pass",  "BasicProbe_pt[passCondition]")
        d = d.Define("Probe_eta_pass", "BasicProbe_eta[passCondition]")
        d = d.Define("TPmass_pass",    "BasicTPmass[passCondition]")
        d = d.Define("Probe_u_pass",        "BasicProbe_u[passCondition]")
        d = d.Define("Probe_charge_pass",   "BasicProbe_charge[passCondition]")
        # fail probes
        d = d.Define("Probe_pt_fail",  "BasicProbe_pt[failCondition]")
        d = d.Define("Probe_eta_fail", "BasicProbe_eta[failCondition]")
        d = d.Define("TPmass_fail",    "BasicTPmass[failCondition]")        
        d = d.Define("Probe_u_fail",        "BasicProbe_u[failCondition]")
        d = d.Define("Probe_charge_fail",   "BasicProbe_charge[failCondition]")

        makeAndSaveHistograms(d, histo_name, "IsolationNoTrigger", binning_mass, binning_pt, binning_eta)

else:
    # for the veto selection
    # chargeCut = ""
    # if args.charge:
    #     sign= ">" if args.charge > 0 else "<"
    #     chargeCut = f" && Muon_charge {sign} 0"
    # use tracks for all probes rather than muons
    if(args.isData == 1):
        d = d.Define("isGenMatchedTrack","createTrues(nTrack)")
    else:
        d = d.Define("isGenMatchedTrack", "hasGenMatch(  GenMuonBare_eta, GenMuonBare_phi, Track_eta, Track_phi)")
        d = d.Define("GenMatchedIdx",     "GenMatchedIdx(GenMuonBare_eta, GenMuonBare_phi, Track_eta, Track_phi)")

    d = d.Define("BasicProbe_Muons", "Track_pt > 10 && abs(Track_eta) < 2.4 && Track_trackOriginalAlgo != 13 && Track_trackOriginalAlgo != 14 && isGenMatchedTrack && (Track_qualityMask & 4)")

    d = d.Define("All_TPPairs", f"CreateTPPair(Tag_Muons, BasicProbe_Muons, {doOS}, Tag_charge, Track_charge, Tag_inExtraIdx, Track_extraIdx)")
    d = d.Define("All_TPmass","getTPmass(All_TPPairs, Tag_pt, Tag_eta, Tag_phi, Track_pt, Track_eta, Track_phi)")
    massLow  =  60
    massHigh = 120
    binning_mass = array('d',[massLow + i for i in range(int(1+massHigh-massLow))])
    massCut = f"All_TPmass > {massLow} && All_TPmass < {massHigh}"

    # overriding previous pt binning
    ## FIXME: to optimize
    binning_pt = array('d',[(10. + 5.*i) for i in range(12)]) # from MC truth the efficiency of the veto is flat versus pt from 20 to 65 GeV

    d = d.Define("TPPairs", f"All_TPPairs[{massCut}]")
    # call it BasicTPmass so it can be filtered later without using Redefine, but an appropriate Define
    d = d.Define("BasicTPmass",  f"All_TPmass[{massCut}]")

    # define all basic probes here (these are all Muon), to be filtered further later, without having to use Redefine when filtering
    d = d.Define("BasicProbe_charge", "getVariables(TPPairs, Track_charge, 2)")
    d = d.Define("BasicProbe_pt",     "getVariables(TPPairs, Track_pt,     2)")
    d = d.Define("BasicProbe_eta",    "getVariables(TPPairs, Track_eta,    2)")

    d = d.Define("vetoMuons", "Muon_pt > 10 && abs(Muon_eta) < 2.4 && (Muon_isGlobal || Muon_isTracker) && Muon_innerTrackOriginalAlgo != 13 && Muon_innerTrackOriginalAlgo != 14 && Muon_looseId && abs(Muon_dxybs) < 0.05 && Muon_highPurity")
    #d = d.Define("passCondition_veto", "coll1coll2DR(Track_eta, Track_phi, Muon_eta[vetoMuons], Muon_phi[vetoMuons]) < 0.1")
    d = d.Define("passCondition_veto",
                 "Probe_isMatched(TPPairs, Track_extraIdx, Muon_innerTrackExtraIdx, vetoMuons)")

    # define condition for passing probes
    d = d.Define("passCondition", "getVariables(TPPairs, passCondition_veto, 2)")
    d = d.Define("failCondition", "!passCondition")            
    # pass probes
    d = d.Define("Probe_pt_pass",  "BasicProbe_pt[passCondition]")
    d = d.Define("Probe_eta_pass", "BasicProbe_eta[passCondition]")
    d = d.Define("TPmass_pass",    "BasicTPmass[passCondition]")
    # fail probes
    d = d.Define("Probe_pt_fail",  "BasicProbe_pt[failCondition]")
    d = d.Define("Probe_eta_fail", "BasicProbe_eta[failCondition]")
    d = d.Define("TPmass_fail",    "BasicTPmass[failCondition]")
    makeAndSaveHistograms(d, histo_name, "veto", binning_mass, binning_pt, binning_eta)
    

        
f_out.Close()

print(d.Report().Print())

elapsed = time.time() - tstart
elapsed_cpu = time.process_time() - cpustrat
print('Execution time:', elapsed, 'seconds')
print('CPU Execution time:', elapsed_cpu , 'seconds')
