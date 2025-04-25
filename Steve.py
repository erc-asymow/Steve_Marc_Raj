import ROOT
from array import array
import json

ROOT.gInterpreter.ProcessLine(".O3") # TODO: can remove this line I think
ROOT.ROOT.EnableImplicitMT()
ROOT.gInterpreter.Declare('#include "Steve.h"')
ROOT.gInterpreter.Declare('#include "GenFunctions.h"')
import os
#from os import listdir
import time
import sys

from runAll import common_parser
from dataset_tools import makeFilelist

import argparse


def makeAndSaveOneHist(d, histo_name, histo_title, binning_mass, binning_pt, binning_eta, massVar="TPmass", isPass=True, scaleFactor=1.0):

    passStr = "pass" if isPass else "fail"
    model = ROOT.RDF.TH3DModel(f"{passStr}_mu_{histo_name}", f"{histo_title} {passStr}",
                               len(binning_mass)-1, binning_mass,
                               len(binning_pt)-1, binning_pt,
                               len(binning_eta)-1, binning_eta)

    histogram = d.Histo3D(model, f"{massVar}_{passStr}", f"Probe_pt_{passStr}", f"Probe_eta_{passStr}", "weight")
    histogram.Scale(scaleFactor)
    histogram.Write()


def makeAndSaveHistograms(d, histo_name, histo_title, binning_mass, binning_pt, binning_eta, massVar="TPmass", scaleFactor=1.0):

    if "dR_probe_gen" in d.GetColumnNames():
        model_deltaR = ROOT.RDF.TH1DModel(f"dR", f"dR(probe,gen)", 320, 0, 5.5)
        deltaR_hist = d.Histo1D(model_deltaR, "dR_probe_gen")
        print(deltaR_hist.Integral())
        deltaR_hist.Write()

    makeAndSaveOneHist(d, histo_name, histo_title, binning_mass, binning_pt, binning_eta, massVar, isPass=True, scaleFactor=scaleFactor)    
    makeAndSaveOneHist(d, histo_name, histo_title, binning_mass, binning_pt, binning_eta, massVar, isPass=False, scaleFactor=scaleFactor)    


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

#parser = argparse.ArgumentParser()
parser = common_parser()

parser.add_argument("-e","--efficiency",
		    help="1 for reco, 2 for 'tracking', 3 for idip, 4 for trigger, 5 for isolation, 6 for isolation without trigger, 7 for isolation with failing trigger, 8 for veto (loose ID+dxybs<0.05) on top of global muons, 9 for P(tracker-seeded track | Standalone muon), 10 for p(tracker muon and not global| tracker-seeded track), 11 for veto on top of 'global or tracker' ",
                    type=int, choices=range(1,12))
parser.add_argument("--testVetoStrategy",
                    default=0,
		    help="Different definition for test veto efficiency measurement (check code for details)",
                    type=int, choices=range(3))

#parser.add_argument("-i","--input_path", help="path of the input root files", type=str)

parser.add_argument("-i","--input_path", nargs='+', default=[], help="path of the input root files")

parser.add_argument("-o","--output_file", help="name of the output root file",
                    type=str)

parser.add_argument("--isData", action='store_true', help="Run on data")

parser.add_argument("-b","--isBkg", action='store_true', help="Run on a background MC process")

parser.add_argument("--genMatchCut", help="Gen-matching cut flag. Use 0 for no genMatching, 1 for default genMatching, -1 for reverse genMatching (useful for Zjets)",
                    type=float, default=1, choices=[-1, 0, 1])

parser.add_argument("-c","--charge", help="Make efficiencies for a specific charge of the probe (-1/1 for positive negative, 0 for inclusive)",
                    type=int, default=0, choices=[-1, 0, 1])

parser.add_argument("-p","--eventParity", help="Select events with given parity for statistical tests, -1/1 for odd/even events, 0 for all (default)",
                    type=int, default=0, choices=[-1, 0, 1])

parser.add_argument("-zqt","--zqtprojection", action="store_true", help="Efficiencies evaluated as a function of zqtprojection (only for trigger and isolation)")

parser.add_argument("-gen","--genLevelEfficiency", action="store_true", help="Compute MC truth efficiency. This needs to be cross-checked as there might be bugs or missing updates.")

parser.add_argument("-tpg","--tnpGenLevel", action="store_true", help="Compute tag-and-probe efficiencies for MC as a function of postVFP gen variables")

parser.add_argument("-nf", "--normFactor", help="Normalization factor for the event weight in MC (can be cross section times luminosity)",
                    type=float, default=1.0)

args = parser.parse_args()
tstart = time.time()
cpustrat = time.process_time()

# compare pt values within some tolerance
if (args.histMinPt + 0.01) < args.innerTrackMinPt:
    raise IOError(f"Inconsistent values for options --histMinPt ({args.histMinPt}) and --innerTrackMinPt ({args.innerTrackMinPt}).\nThe former must not be smaller than the latter.\n")

if args.isData & args.genLevelEfficiency:
    raise RuntimeError('\'genLevelEfficiency\' option not supported for data')

if args.isData & args.tnpGenLevel:
    raise RuntimeError('\'tnpGenLevel\' option not supported for data')

if not args.output_file.endswith(".root"):
    raise NameError('output_file name must end with \'.root\'')

# create output folders if not existing
outdir = os.path.dirname(os.path.abspath(args.output_file))
if not os.path.exists(outdir):
    print()
    print(f"Creating folder {outdir} to store outputs")
    os.makedirs(outdir)    
    print()

if args.isData:
    histo_name= "RunGtoH"
else:
    histo_name = "DY_postVFP"

files=[]

# print(args.input_path)
files = makeFilelist(args.input_path, maxFiles=args.maxFiles)
# print(files)

if args.charge and args.efficiency in [2, 9]:
    print("")
    print("   WARNING: charge splitting for tracking efficiency is implemented using the tag muon (with the other charge)")
    print("")


filenames = ROOT.std.vector('string')()

for name in files: filenames.push_back(name)

d = ROOT.RDataFrame("Events", filenames)

# had to hack, since definition of the vertex variables were not consistent throughout 
# various productions. Made alias of the new variables since the part where actual calculation 
# is done remains same
d = d.Alias("Muon_Z", "Muon_vz")
d = d.Alias("Muon_X", "Muon_vx")
d = d.Alias("Muon_Y", "Muon_vy")
d = d.Alias("Track_Z", "Track_vz")
d = d.Alias("Track_X", "Track_vx")
d = d.Alias("Track_Y", "Track_vy")

masshighut = 200

minStandaloneNumberOfValidHits = args.standaloneValidHits
minStandalonePt = args.standaloneMinPt
minInnerTrackPt = args.innerTrackMinPt
minHistPt = args.histMinPt

default_pt_binning = [10., 15., 20., 24., 26., 28., 30., 32., 34., 36., 38., 40., 42., 44., 47., 50., 55., 60., 65.]
default_pt_binning = [x for x in default_pt_binning if (x+0.1) >= minHistPt]

binning_pt = array('d', default_pt_binning)
binning_eta = array('d',[round(-2.4 + i*0.1,2) for i in range(49)])
binning_mass = array('d',[60 + i for i in range(masshighut-60+1)])
binning_charge = array('d',[-1.5,0,1.5])
binning_u = array('d',[-3000000000,-30,-15,-10,-5,0,5,10,15,30,40,50,60,70,80,90,100,30000000000])

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

if(args.isData):
    d = d.Define("gen_weight","1")
else:
    d = d.Define("gen_weight", "clipGenWeight(genWeight)") #for now (testing)
weightSum = d.Sum("gen_weight")

if not args.genLevelEfficiency:
    ##General Cuts
    d = d.Filter("PV_npvsGood >= 1","NVtx Cut")
    if(args.year == "2016"):
        d = d.Filter("HLT_IsoMu24 || HLT_IsoTkMu24", "HLT Cut")
    else:
        d = d.Filter("HLT_IsoMu24", "HLT Cut")

# for statistical tests (postfix to be added to file name)
if args.eventParity < 0:
    d = d.Filter("(event % 2) == 1") # odd
elif args.eventParity > 0:
    d = d.Filter("(event % 2) == 0") # even

doOS = 0 if args.noOppositeCharge else 1
doOStracking = 0 if args.noOppositeChargeTracking else doOS

doSameCharge = 1 if args.SameCharge else 0

if doOS & doSameCharge:
    raise Exception("Both doOS and doSameCharge can't be True. Require noOppositeCharge if you really want the Same Sign")

if doOStracking & doSameCharge:
    raise Exception("Both doOStracking and doSameCharge can't be True. Require noOppositeChargeTracking if you really want Same Sign")

if (args.isData):
    if (args.year == "2016"): 
        jsonhelper = make_jsonhelper("Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")
    elif (args.year == "2018"):
        jsonhelper = make_jsonhelper("utility/Cert_314472-325175_13TeV_UL2018_Collisions18_HLT_IsoMu24_v_CustomJSON.txt")
    elif (args.year == "2017"):
        print("2017 is not yet supported")
        quit()
    d = d.Filter(jsonhelper,["run","luminosityBlock"],"jsonfilter")

## Weights

if (args.isData):
    d = d.Define("weight","1")
else:
    if not args.noVertexPileupWeight:
        if hasattr(ROOT, "initializeVertexPileupWeights"):
            print("Initializing histograms with vertex-pileup weights")
            input_vertexWeight = ""
            if (args.year == "2016"):
                input_vertexWeight = "./utility/vertexPileupWeights.root"
            elif (args.year == "2017"):
                input_vertexWeight = "./utility/vtx_reweight_2dPUandbeamspot2017.root"
            elif (args.year == "2018"):
                input_vertexWeight = "./utility/vtx_reweight_2dPUandbeamspot2018.root"
            ROOT.initializeVertexPileupWeights(input_vertexWeight,args.year)
            d = d.Define("vertex_weight", "_get_vertexPileupWeight(GenVtx_z,Pileup_nTrueInt,2)")
    else:
        d = d.Define("vertex_weight", "1.0")
    if (args.year == "2016"):
        d = d.Define("pu_weight", "puw_2016(Pileup_nTrueInt,2)") # 2 is for postVFP
    else:
        if hasattr(ROOT, "initializePileupWeights"):
            print("Initializing Pileup weights for 2017 or 2018")
            input_PU_mc = ""
            input_PU_data = ""
            if(args.year == "2017"):
                input_PU_mc = "./utility/MC2017PU.root"
                input_PU_data = "./utility/pileupHistogram-customJSON-UL2017-69200ub-99bins.root"
            elif(args.year == "2018"):
                input_PU_mc = "./utility/MC2018PU.root"
                input_PU_data = "./utility/pileupHistogram-customJSON-UL2018-69200ub-99bins.root"
            ROOT.initializePileupWeights(input_PU_mc,input_PU_data)
            d = d.Define("pu_weight", "_get_PileupWeight(Pileup_nTrueInt,2)")
            #d = d.Define("pu_weight", "puw_2016(Pileup_nTrueInt,2)") # 2 is for postVFP
 
    d = d.Define("weight", "gen_weight*pu_weight*vertex_weight")

## For Tag Muons
if args.year == "2016":
    d = d.Define("isTriggeredMuon","hasTriggerMatch(Muon_eta, Muon_phi, TrigObj_id, TrigObj_filterBits, TrigObj_eta, TrigObj_phi)")
else:
    d =d.Define("isTriggeredMuon","hasTriggerMatch2018(Muon_eta, Muon_phi, TrigObj_id, TrigObj_filterBits, TrigObj_eta, TrigObj_phi)")

if(args.isData):
    d = d.Define("isGenMatchedMuon","createTrues(nMuon)")
else: 
    d = d.Define("GenMuonBare", "GenPart_status == 1 && (GenPart_statusFlags & 1 || GenPart_statusFlags & (1<<5)) && abs(GenPart_pdgId) == 13")
    d = d.Define("GenMuonBare_pt", "GenPart_pt[GenMuonBare]")
    d = d.Define("GenMuonBare_eta", "GenPart_eta[GenMuonBare]")
    d = d.Define("GenMuonBare_phi", "GenPart_phi[GenMuonBare]")
    d = d.Define("GenMuonBare_pdgId", "GenPart_pdgId[GenMuonBare]")
    d = d.Define("isGenMatchedMuon", "hasGenMatch(GenMuonBare_eta, GenMuonBare_phi, Muon_eta, Muon_phi)")

## Define tags as trigger matched and gen matched (gen match can be removed with an option in case)
#
# just for utility
d = d.Alias("Tag_pt",  "Muon_pt")
d = d.Alias("Tag_eta", "Muon_eta")
d = d.Alias("Tag_phi", "Muon_phi")
d = d.Alias("Tag_charge", "Muon_charge")
d = d.Alias("Tag_Z", "Muon_vz") # for tag-probe Z difference cut 
d = d.Alias("Tag_inExtraIdx", "Muon_innerTrackExtraIdx")
d = d.Alias("Tag_outExtraIdx", "Muon_standaloneExtraIdx")
# for tracking we may want to test efficiencies by charge, but in that case we enforce the (other) charge on the tag
# under the assumption that tag and probe muons have opposite charge (but we still don't force opposite charge explicitly)
TagAntiChargeCut = ""
if args.efficiency in [2, 9] and args.charge:
    TagAntiChargeCut = " && Tag_charge < 0" if args.charge > 0 else " && Tag_charge > 0" # note that we swap charge 
# now define the tag muon (Muon_isGlobal might not be necessary, but shouldn't hurt really)
d = d.Define("Tag_Muons", f"Muon_pt > {args.tagPt} && abs(Muon_eta) < 2.4 && Muon_pfRelIso04_all < {args.tagIso} && abs(Muon_dxybs) < 0.05 && Muon_mediumId && Muon_isGlobal && isTriggeredMuon && isGenMatchedMuon {TagAntiChargeCut}")

if not args.genLevelEfficiency:
    d = d.Filter("Sum(Tag_Muons) > 0") # would this make the loop a little faster?

if (args.genLevelEfficiency):
    d = d.DefinePerSample("zero","0").DefinePerSample("one","1")
    d = d.Define("GenPart_postFSRLepIdx1","PostFSRIdx(GenPart_pdgId,GenPart_status,GenPart_genPartIdxMother,GenPart_statusFlags,GenPart_pt,zero)")
    d = d.Define("GenPart_postFSRLepIdx2","PostFSRIdx(GenPart_pdgId,GenPart_status,GenPart_genPartIdxMother,GenPart_statusFlags,GenPart_pt,one)")
    d = d.Define("goodgenpt","goodgenvalue(GenPart_pt,GenPart_postFSRLepIdx1,GenPart_postFSRLepIdx2,GenPart_eta,GenPart_phi,GenPart_status,GenPart_pdgId)")
    d = d.Define("goodgeneta","goodgenvalue(GenPart_eta,GenPart_postFSRLepIdx1,GenPart_postFSRLepIdx2,GenPart_eta,GenPart_phi,GenPart_status,GenPart_pdgId)")
    d = d.Define("goodgenphi","goodgenvalue(GenPart_phi,GenPart_postFSRLepIdx1,GenPart_postFSRLepIdx2,GenPart_eta,GenPart_phi,GenPart_status,GenPart_pdgId)")
    d = d.Define("goodgencharge","goodgencharge(GenPart_postFSRLepIdx1,GenPart_postFSRLepIdx2,GenPart_eta,GenPart_phi,GenPart_status,GenPart_pdgId)")
    d = d.Define("goodgenidx","goodgenidx(GenPart_pt,GenPart_postFSRLepIdx1,GenPart_postFSRLepIdx2,GenPart_eta,GenPart_phi,GenPart_status,GenPart_pdgId)")
    d = d.Define("postFSRgenzqtprojection","postFSRgenzqtprojection(goodgenpt,goodgeneta,goodgenphi)")
    if args.charge:
        sign= ">" if args.charge > 0 else "<"
        d = d.Redefine("goodgenpt",f"goodgenpt[goodgencharge {sign} 0]")
        d = d.Redefine("goodgeneta",f"goodgeneta[goodgencharge {sign} 0]")
        d = d.Redefine("goodgenphi",f"goodgenphi[goodgencharge {sign} 0]")
        d = d.Redefine("goodgenidx",f"goodgenidx[goodgencharge {sign} 0]")
        d = d.Redefine("postFSRgenzqtprojection",f"postFSRgenzqtprojection[goodgencharge {sign} 0]")
        d = d.Redefine("goodgencharge",f"goodgencharge[goodgencharge {sign} 0]")
    # this might be done simply as
    # d = d.Define("goodgenpt", "GenMuonBare_pt") # the collection might have more than 2 elements here, but can be easily filtered (should be sorted too)


# Open output file
f_out = ROOT.TFile(args.output_file, "RECREATE")

## Tracks for reco efficiency
if args.efficiency==1:
    if not (args.genLevelEfficiency):

        if (args.isData) or (args.genMatchCut==0):
            d = d.Define("isGenMatchedTrack","createTrues(nTrack)")
        else:
            genMatchCut = 0.1*args.genMatchCut
            d = d.Define("isGenMatchedTrack", f"hasGenMatch(  GenMuonBare_eta, GenMuonBare_phi, Track_eta, Track_phi, {genMatchCut})")
            d = d.Define("GenMatchedIdx",     f"GenMatchedIdx(GenMuonBare_eta, GenMuonBare_phi, Track_eta, Track_phi, {genMatchCut})")

        chargeCut = ""
        if args.charge:
            sign= ">" if args.charge > 0 else "<"
            chargeCut = f" && Track_charge {sign} 0"

        # define all probes
        d = d.Define("Probe_Tracks", f"Track_pt > {minInnerTrackPt} && abs(Track_eta) < 2.4 && Track_trackOriginalAlgo != 13 && Track_trackOriginalAlgo != 14 && isGenMatchedTrack && (Track_qualityMask & 4) {chargeCut}")
        # condition for passing probes
        # FIXME: add other criteria to the MergedStandAloneMuon to accept the matching? E.g. |eta| < 2.4? No, it is an acceptance on numerator which we don't want
        d = d.Define("goodStandaloneMuon", f"MergedStandAloneMuon_pt > {minStandalonePt} && MergedStandAloneMuon_numberOfValidHits >= {minStandaloneNumberOfValidHits}")
        d = d.Define("passCondition_reco", "trackStandaloneDR(Track_eta, Track_phi, MergedStandAloneMuon_eta[goodStandaloneMuon], MergedStandAloneMuon_phi[goodStandaloneMuon]) < 0.3")
        
        d = d.Define("All_TPPairs", f"CreateTPPair(Tag_Muons, Probe_Tracks, {doOS}, Tag_charge, Track_charge, Tag_inExtraIdx, Track_extraIdx, 0, {doSameCharge})")
        d = d.Define("All_TPmass", "getTPmass(All_TPPairs, Tag_pt, Tag_eta, Tag_phi, Track_pt, Track_eta, Track_phi)")
        d = d.Define("All_absDiffZ", "getTPabsDiffZ(All_TPPairs, Tag_Z, Track_Z)")
        
        # overriding previous pt binning
        default_pt_binning = [10., 15., 20., 24., 26., 30., 34., 38., 42., 46., 50., 55., 60., 65.]
        default_pt_binning = [x for x in default_pt_binning if (x+0.1) >= minHistPt]
        binning_pt = array('d',default_pt_binning)
        ## binning is currently 50,130 GeV, but it is overridden below 
        # also for mass
        massLow  =  60
        massHigh = 120
        binning_mass = array('d',[massLow + i for i in range(int(1+massHigh-massLow))])
        massCut = f"All_TPmass > {massLow} && All_TPmass < {massHigh}"
        ZdiffCut = "All_absDiffZ < 0.2"
        d = d.Define("TPPairs", f"All_TPPairs[{massCut} && {ZdiffCut}]")
        d = d.Define("TPmass",  f"All_TPmass[{massCut}  && {ZdiffCut}]")

        d = d.Define("Probe_pt",  "getVariables(TPPairs, Track_pt,  2)")
        d = d.Define("Probe_eta", "getVariables(TPPairs, Track_eta, 2)")
        d = d.Define("Probe_phi", "getVariables(TPPairs, Track_phi, 2)")
        
        d = d.Define("passCondition", "getVariables(TPPairs, passCondition_reco, 2)")
        d = d.Define("failCondition", "!passCondition")

        if (args.tnpGenLevel):
            d = d.Redefine("Probe_pt",  "getGenVariables(TPPairs, GenMatchedIdx, GenMuonBare_pt,  2)")
            d = d.Redefine("Probe_eta", "getGenVariables(TPPairs, GenMatchedIdx, GenMuonBare_eta, 2)")
            d = d.Redefine("Probe_phi", "getGenVariables(TPPairs, GenMatchedIdx, GenMuonBare_phi, 2)")
        
        d = d.Define("Probe_pt_pass",  "Probe_pt[passCondition]")
        d = d.Define("Probe_eta_pass", "Probe_eta[passCondition]")
        d = d.Define("TPmass_pass", "TPmass[passCondition]")
        d = d.Define("Probe_pt_fail",  "Probe_pt[failCondition]")
        d = d.Define("Probe_eta_fail", "Probe_eta[failCondition]")
        d = d.Define("TPmass_fail", "TPmass[failCondition]")

        if not args.isData:  #saving the distribution of the dR between the gen muon and the probe, to have an a-posteriori check of the choice
            d = d.Define("dR_probe_gen", "coll1coll2DR(Probe_eta, Probe_phi, GenMuonBare_eta, GenMuonBare_phi)")

        normFactor = args.normFactor
        scale = 1.0 if args.isData else (normFactor/weightSum.GetValue()) if args.normalizeMCsumGenWeights else normFactor
        makeAndSaveHistograms(d, histo_name, "Reco", binning_mass, binning_pt, binning_eta, scaleFactor=scale)

    else:
        d = d.Define("goodmuon","trackStandaloneDR(goodgeneta, goodgenphi, MergedStandAloneMuon_eta[goodStandaloneMuon], MergedStandAloneMuon_phi[goodStandaloneMuon]) < 0.3").Define("newweight","weight*goodmuon")

        model_pass_reco = ROOT.RDF.TH2DModel("Pass","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)
        model_norm_reco = ROOT.RDF.TH2DModel("Norm","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)

        pass_histogram_reco = d.Histo2D(model_pass_reco,"goodgeneta","goodgenpt","newweight")
        pass_histogram_norm = d.Histo2D(model_norm_reco,"goodgeneta","goodgenpt","weight")

        pass_histogram_reco.Write()
        pass_histogram_norm.Write()


#Global|MergedStandAloneMuon ("tracking" efficiency)
elif args.efficiency==2:
    if not (args.genLevelEfficiency):

        if (args.isData) or (args.genMatchCut==0):
            d = d.Define("isGenMatchedMergedStandMuon", "createTrues(nMergedStandAloneMuon)")
        else:
            genMatchCut = 0.3*args.genMatchCut
            d = d.Define("isGenMatchedMergedStandMuon", f"hasGenMatch(GenMuonBare_eta, GenMuonBare_phi, MergedStandAloneMuon_eta, MergedStandAloneMuon_phi, {genMatchCut})")
            d = d.Define("GenMatchedIdx", f"GenMatchedIdx(GenMuonBare_eta, GenMuonBare_phi, MergedStandAloneMuon_eta, MergedStandAloneMuon_phi, {genMatchCut})")

        # All probes, standalone muons from MergedStandAloneMuon_XX    
        d = d.Define("Probe_MergedStandMuons",f"MergedStandAloneMuon_pt > {minStandalonePt} && abs(MergedStandAloneMuon_eta) < 2.4 && isGenMatchedMergedStandMuon && MergedStandAloneMuon_numberOfValidHits >= {minStandaloneNumberOfValidHits}")
        #
        d = d.Define("All_TPPairs", f"CreateTPPair(Tag_Muons, Probe_MergedStandMuons, {doOStracking}, Tag_charge, MergedStandAloneMuon_charge, Tag_outExtraIdx, MergedStandAloneMuon_extraIdx, 0, {doSameCharge})")
        d = d.Define("All_TPmass","getTPmass(All_TPPairs, Tag_pt, Tag_eta, Tag_phi, MergedStandAloneMuon_pt, MergedStandAloneMuon_eta, MergedStandAloneMuon_phi)")

        massLow  =  50
        massHigh = 130
        binning_mass = array('d',[massLow + i for i in range(int(1+massHigh-massLow))])
        massCut = f"All_TPmass > {massLow} && All_TPmass < {massHigh}"
        d = d.Define("TPPairs", f"All_TPPairs[{massCut}]")
        d = d.Define("TPmass",  f"All_TPmass[{massCut}]")

        d = d.Define("Probe_pt",  "getVariables(TPPairs, MergedStandAloneMuon_pt,  2)")
        d = d.Define("Probe_eta", "getVariables(TPPairs, MergedStandAloneMuon_eta, 2)")
        d = d.Define("Probe_phi", "getVariables(TPPairs, MergedStandAloneMuon_phi, 2)")

        if (args.tnpGenLevel):
            d = d.Redefine("Probe_pt",  "getGenVariables(TPPairs, GenMatchedIdx, GenMuonBare_pt,  2)")
            d = d.Redefine("Probe_eta", "getGenVariables(TPPairs, GenMatchedIdx, GenMuonBare_eta, 2)")
            d = d.Redefine("Probe_eta", "getGenVariables(TPPairs, GenMatchedIdx, GenMuonBare_phi, 2)")

        # condition for passing probe, start from Muon_XX and then add match of extraID indices between Muon and MergedStandAloneMuon
        #d = d.Define("Muon_standaloneNvalidHits", "getGlobalMuon_MergedStandAloneMuonVar(Muon_standaloneExtraIdx, MergedStandAloneMuon_extraIdx, MergedStandAloneMuon_numberOfValidHits)")
        d = d.Define("Muon_forTracking", f"Muon_isGlobal && Muon_pt > {minInnerTrackPt} && Muon_standalonePt > {minStandalonePt} && selfDeltaR(Muon_eta, Muon_phi, Muon_standaloneEta, Muon_standalonePhi) < 0.3 && Muon_highPurity")
        ## && Muon_standaloneNvalidHits >= {minStandaloneNumberOfValidHits}
        # 
        # check Muon exists with proper criteria and matching extraIdx with the standalone muon 
        # Note: no need to enforce the number of valid hits of the standalone track for the global muon,
        ##      since it is already embedded in the denominator
        ##      and the matching already ensures it is propagated to the numerator
        d = d.Define("passCondition_tracking",
                     "Probe_isMatched(TPPairs, MergedStandAloneMuon_extraIdx, Muon_standaloneExtraIdx, Muon_forTracking)")
        # the following was equivalent but with an additional check between probe and tag, to exclude having the same inner track, it should not be necessary
        #d = d.Define("passCondition_tracking",
        #             "Probe_isGlobal_checkExtraIdxTagInnerTrack(TPPairs, MergedStandAloneMuon_extraIdx, Muon_standaloneExtraIdx, Muon_forTracking, Tag_inExtraIdx, Muon_innerTrackExtraIdx)")
        d = d.Define("passCondition", "getVariables(TPPairs, passCondition_tracking, 2)")
        d = d.Define("failCondition", "!passCondition")

        # the binning should match the standalone pt (usually 15),
        # but from 10 to 15 the efficiency is probably impossible to measure, so this bin will just remain empty
        # and should be filled with the content of the bin immediately above it afterwards at analysis level if needed
        default_pt_binning = [10., 15., 24., 35., 45., 55., 65.]
        default_pt_binning = [x for x in default_pt_binning if (x+0.1) >= minHistPt]
        binning_pt = array('d', default_pt_binning)

        # Here we are using the muon variables to calulate the mass for the passing probes for tracking efficiency
        ## However the TPPairs were made using indices from MergedStandAloneMuon_XX collections, which are not necessarily valid for Muon_XX
        ## Thus, for each passing MergedStandAloneMuon I store the pt,eta,phi of the corresponding Muon (which exists as long as we use the MergedStandAloneMuon indices from TPPairs_pass)
        d = d.Define("TPPairs_pass", "TPPairs[passCondition]")
        d = d.Define("MergedStandaloneMuon_MuonIdx", "getMergedStandAloneMuon_MuonIdx(MergedStandAloneMuon_extraIdx, Muon_standaloneExtraIdx)")
        d = d.Define("MergedStandaloneMuon_MuonPt",  "getMergedStandAloneMuon_matchedObjectVar(MergedStandaloneMuon_MuonIdx, Muon_pt)")
        d = d.Define("MergedStandaloneMuon_MuonEta", "getMergedStandAloneMuon_matchedObjectVar(MergedStandaloneMuon_MuonIdx, Muon_eta)")
        d = d.Define("MergedStandaloneMuon_MuonPhi", "getMergedStandAloneMuon_matchedObjectVar(MergedStandaloneMuon_MuonIdx, Muon_phi)")

        d = d.Define("TPmass_pass",    "getTPmass(TPPairs_pass, Tag_pt, Tag_eta, Tag_phi, MergedStandaloneMuon_MuonPt, MergedStandaloneMuon_MuonEta, MergedStandaloneMuon_MuonPhi)")
        d = d.Define("Probe_pt_pass",  "Probe_pt[passCondition]")
        d = d.Define("Probe_eta_pass", "Probe_eta[passCondition]")

        d = d.Define("TPmass_fail",    "TPmass[failCondition]")
        d = d.Define("Probe_pt_fail",  "Probe_pt[failCondition]")
        d = d.Define("Probe_eta_fail", "Probe_eta[failCondition]")

        if not args.isData:  #saving the distribution of the dR between the gen muon and the probe, to have an a-posteriori check of the choice
            d = d.Define("dR_probe_gen", "coll1coll2DR(Probe_eta, Probe_phi, GenMuonBare_eta, GenMuonBare_phi)")

        normFactor = args.normFactor
        scale = 1.0 if args.isData else (normFactor/weightSum.GetValue()) if args.normalizeMCsumGenWeights else normFactor
        makeAndSaveHistograms(d, histo_name, "Tracking", binning_mass, binning_pt, binning_eta, scaleFactor=scale)

        # save also the mass for passing probes computed with standalone variables
        # needed when making MC template for failing probes using all probes, since the mass should be consistently measured for both cases
        # do it also for data in case we want to check the difference in the efficiencies
        d = d.Define("TPmassFromSA_pass", "TPmass[passCondition]")
        makeAndSaveOneHist(d, f"{histo_name}_alt", "Tracking (mass from SA muons)",
                           binning_mass, binning_pt, binning_eta,
                           massVar="TPmassFromSA", isPass=True, scaleFactor=scale)
        
    else:
        d = d.Define("Muon_standaloneNvalidHits", "getGlobalMuon_MergedStandAloneMuonVar(Muon_standaloneExtraIdx, MergedStandAloneMuon_extraIdx, MergedStandAloneMuon_numberOfValidHits)")
        d = d.Define("passHasHits",f"Muon_standaloneNvalidHits >= {minStandaloneNumberOfValidHits}")
        d = d.Define("Muon_forTrackingWithHits","Muon_forTracking && passHasHits")
        d = d.Define("goodmuon","trackStandaloneDR(goodgeneta, goodgenphi, Muon_eta[Muon_forTrackingWithHits], Muon_phi[Muon_forTrackingWithHits]) < 0.3").Define("newweight","weight*goodmuon")

        model_pass_reco = ROOT.RDF.TH2DModel("Pass","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)
        model_norm_reco = ROOT.RDF.TH2DModel("Norm","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)

        pass_histogram_reco = d.Histo2D(model_pass_reco,"goodgeneta","goodgenpt","newweight")
        pass_histogram_norm = d.Histo2D(model_norm_reco,"goodgeneta","goodgenpt","weight")

        pass_histogram_reco.Write()
        pass_histogram_norm.Write()

## Muons for all other nominal efficiency steps including veto on top of tracking
elif args.efficiency < 9:
    
    if not args.isData:
        d = d.Define("GenMatchedIdx","GenMatchedIdx(GenMuonBare_eta, GenMuonBare_phi, Muon_eta, Muon_phi)")

    chargeCut = ""
    if args.charge:
        sign= ">" if args.charge > 0 else "<"
        chargeCut = f" && Muon_charge {sign} 0"
        
    d = d.Define("Muon_standaloneNvalidHits", "getGlobalMuon_MergedStandAloneMuonVar(Muon_standaloneExtraIdx, MergedStandAloneMuon_extraIdx, MergedStandAloneMuon_numberOfValidHits)")
    d = d.Define("BasicProbe_Muons", f"Muon_isGlobal && Muon_pt > {minInnerTrackPt} && Muon_standalonePt > {minStandalonePt} && abs(Muon_eta) < 2.4 && selfDeltaR(Muon_eta, Muon_phi, Muon_standaloneEta, Muon_standalonePhi) < 0.3 && Muon_highPurity && isGenMatchedMuon {chargeCut} && Muon_standaloneNvalidHits >= {minStandaloneNumberOfValidHits}")
    # add MergedStandAloneMuon_numberOfValidHits > 0 matching the global muon to its standalone one 
    d = d.Define("All_TPPairs", f"CreateTPPair(Tag_Muons, BasicProbe_Muons, {doOS}, Tag_charge, Muon_charge, Tag_inExtraIdx, Muon_innerTrackExtraIdx, 1, {doSameCharge})") # these are all Muon_XX, so might just exclude same index in the loop
    d = d.Define("All_TPmass","getTPmass(All_TPPairs, Tag_pt, Tag_eta, Tag_phi, Muon_pt, Muon_eta, Muon_phi)")

    massLow  =  60
    if args.zqtprojection:
        massHigh = masshighut #BE CAREFUL HERE, THIS IS FOR FITS AS A FUNCTION OF UT, RANGE IS NARROWER FOR UT INTEGRATED FITS (massHigh = 120)
    else:
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
    d = d.Define("passCondition_veto", "Muon_looseId && abs(Muon_dxybs) < 0.05")
    d = d.Define("passCondition_IDIP", "Muon_mediumId && abs(Muon_dxybs) < 0.05")
    d = d.Define("passCondition_Trig", "isTriggeredMuon")
    #d = d.Define("passCondition_Iso",  "Muon_pfRelIso04_all < 0.15") #FOR NOW OLD ISO DEFINITION. OTHER COMMENTED LINES HAVE NEW DEFITION (BOTH CHARGED AND INCLUSIVE)
    if(args.isoDefinition == 1):
        d = d.Define("passCondition_Iso", "Muon_vtxAgnPfRelIso04_all < 0.15")
    elif(args.isoDefinition == 0):
        d = d.Define("passCondition_Iso",  "Muon_pfRelIso04_all < 0.15")
    #d = d.Define("passCondition_Iso", "Muon_vtxAgnPfRelIso04_chg < 0.07")
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
            normFactor = args.normFactor
            scale = 1.0 if args.isData else (normFactor/weightSum.GetValue()) if args.normalizeMCsumGenWeights else normFactor
            makeAndSaveHistograms(d, histo_name, "IDIP", binning_mass, binning_pt, binning_eta, scaleFactor=scale)
        else:
            d = d.Define("Muon_forTracking", f"Muon_isGlobal && Muon_pt > {minInnerTrackPt} && Muon_standalonePt > {minStandalonePt} && selfDeltaR(Muon_eta, Muon_phi, Muon_standaloneEta, Muon_standalonePhi) < 0.3 && Muon_highPurity")
            d = d.Define("Muon_standaloneNvalidHits", "getGlobalMuon_MergedStandAloneMuonVar(Muon_standaloneExtraIdx, MergedStandAloneMuon_extraIdx, MergedStandAloneMuon_numberOfValidHits)")
            d = d.Define("passHasHits",f"Muon_standaloneNvalidHits >= {minStandaloneNumberOfValidHits}")
            d = d.Define("Muon_forTrackingWithHits","Muon_forTracking && passHasHits")
            d = d.Define("Muon_forIDIP", "Muon_forTrackingWithHits && passCondition_IDIP")
            d = d.Define("goodmuon","trackStandaloneDR(goodgeneta, goodgenphi, Muon_eta[Muon_forIDIP], Muon_phi[Muon_forIDIP]) < 0.3")
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

                d = d.Define("Muon_forTracking", f"Muon_isGlobal && Muon_pt > {minInnerTrackPt} && Muon_standalonePt > {minStandalonePt} && selfDeltaR(Muon_eta, Muon_phi, Muon_standaloneEta, Muon_standalonePhi) < 0.3 && Muon_highPurity")
                d = d.Define("Muon_standaloneNvalidHits", "getGlobalMuon_MergedStandAloneMuonVar(Muon_standaloneExtraIdx, MergedStandAloneMuon_extraIdx, MergedStandAloneMuon_numberOfValidHits)")
                d = d.Define("passHasHits",f"Muon_standaloneNvalidHits >= {minStandaloneNumberOfValidHits}")
                d = d.Define("Muon_forTrackingWithHits","Muon_forTracking && passHasHits")
                d = d.Define("Muon_forTrigger", "Muon_forTrackingWithHits && passCondition_IDIP && passCondition_Trig")
                d = d.Define("goodmuon","trackStandaloneDR(goodgeneta, goodgenphi, Muon_eta[Muon_forTrigger], Muon_phi[Muon_forTrigger]) < 0.3")
                d = d.Define("newweight","weight*goodmuon")
                pass_histogram_reco = d.Filter("goodgenpt.size()>=2").HistoND(model_pass_trig,strings_pass)
                pass_histogram_norm = d.Filter("goodgenpt.size()>=2").HistoND(model_norm_trig,strings_norm)

                ROOT.saveHistogramsGen(pass_histogram_reco,pass_histogram_norm,ROOT.std.string(args.output_file))
                

        else:
            if not (args.genLevelEfficiency):
                normFactor = args.normFactor
                scale = 1.0 if args.isData else (normFactor/weightSum.GetValue()) if args.normalizeMCsumGenWeights else normFactor
                makeAndSaveHistograms(d, histo_name, "Trigger", binning_mass, binning_pt, binning_eta, scaleFactor=scale)
            else:
                d = d.Define("Muon_forTracking", f"Muon_isGlobal && Muon_pt > {minInnerTrackPt} && Muon_standalonePt > {minStandalonePt} && selfDeltaR(Muon_eta, Muon_phi, Muon_standaloneEta, Muon_standalonePhi) < 0.3 && Muon_highPurity")
                d = d.Define("Muon_standaloneNvalidHits", "getGlobalMuon_MergedStandAloneMuonVar(Muon_standaloneExtraIdx, MergedStandAloneMuon_extraIdx, MergedStandAloneMuon_numberOfValidHits)")
                d = d.Define("passHasHits",f"Muon_standaloneNvalidHits >= {minStandaloneNumberOfValidHits}")
                d = d.Define("Muon_forTrackingWithHits","Muon_forTracking && passHasHits")
                d = d.Define("Muon_forTrigger", "Muon_forTrackingWithHits && passCondition_IDIP && passCondition_Trig")
                d = d.Define("goodmuon","trackStandaloneDR(goodgeneta, goodgenphi, Muon_eta[Muon_forTrigger], Muon_phi[Muon_forTrigger]) < 0.3")
                d = d.Define("newweight","weight*goodmuon")
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

                d = d.Define("Muon_forTracking", f"Muon_isGlobal && Muon_pt > {minInnerTrackPt} && Muon_standalonePt > {minStandalonePt} && selfDeltaR(Muon_eta, Muon_phi, Muon_standaloneEta, Muon_standalonePhi) < 0.3 && Muon_highPurity")
                d = d.Define("Muon_standaloneNvalidHits", "getGlobalMuon_MergedStandAloneMuonVar(Muon_standaloneExtraIdx, MergedStandAloneMuon_extraIdx, MergedStandAloneMuon_numberOfValidHits)")
                d = d.Define("passHasHits",f"Muon_standaloneNvalidHits >= {minStandaloneNumberOfValidHits}")
                d = d.Define("Muon_forTrackingWithHits","Muon_forTracking && passHasHits")
                d = d.Define("Muon_forIso", "Muon_forTrackingWithHits && passCondition_IDIP && passCondition_Trig && passCondition_Iso")
                d = d.Define("goodmuon","trackStandaloneDR(goodgeneta, goodgenphi, Muon_eta[Muon_forIso], Muon_phi[Muon_forIso]) < 0.3")
                d = d.Define("newweight","weight*goodmuon")
                pass_histogram_reco = d.Filter("goodgenpt.size()>=2").HistoND(model_pass_trig,strings_pass)
                pass_histogram_norm = d.Filter("goodgenpt.size()>=2").HistoND(model_norm_trig,strings_norm)

                ROOT.saveHistogramsGen(pass_histogram_reco,pass_histogram_norm,ROOT.std.string(args.output_file))

        else:
            if not (args.genLevelEfficiency):
                normFactor = args.normFactor
                scale = 1.0 if args.isData else (normFactor/weightSum.GetValue()) if args.normalizeMCsumGenWeights else normFactor
                makeAndSaveHistograms(d, histo_name, "Isolation", binning_mass, binning_pt, binning_eta, scaleFactor=scale)
            else:
                d = d.Define("Muon_forTracking", f"Muon_isGlobal && Muon_pt > {minInnerTrackPt} && Muon_standalonePt > {minStandalonePt} && selfDeltaR(Muon_eta, Muon_phi, Muon_standaloneEta, Muon_standalonePhi) < 0.3 && Muon_highPurity")
                d = d.Define("Muon_standaloneNvalidHits", "getGlobalMuon_MergedStandAloneMuonVar(Muon_standaloneExtraIdx, MergedStandAloneMuon_extraIdx, MergedStandAloneMuon_numberOfValidHits)")
                d = d.Define("passHasHits",f"Muon_standaloneNvalidHits >= {minStandaloneNumberOfValidHits}")
                d = d.Define("Muon_forTrackingWithHits","Muon_forTracking && passHasHits")
                d = d.Define("Muon_forIso", "Muon_forTrackingWithHits && passCondition_IDIP && passCondition_Trig && passCondition_Iso")
                d = d.Define("goodmuon","trackStandaloneDR(goodgeneta, goodgenphi, Muon_eta[Muon_forIso], Muon_phi[Muon_forIso]) < 0.3")
                d = d.Define("newweight","weight*goodmuon")
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

        if (args.zqtprojection):
            if not (args.genLevelEfficiency):
                model_pass_iso = ROOT.RDF.THnDModel("pass_mu_"+histo_name, "IsolationNoTrigger_pass", 5, NBIN, XBINS)
                model_fail_iso = ROOT.RDF.THnDModel("fail_mu_"+histo_name, "IsolationNoTrigger_fail", 5, NBIN, XBINS)
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
                model_pass_trig = ROOT.RDF.THnDModel("pass_mu_"+histo_name, "IsolationNoTrigger_pass", 4, GENNBIN, GENXBINS)
                model_norm_trig = ROOT.RDF.THnDModel("norm_mu_"+histo_name, "IsolationNoTrigger_norm", 4, GENNBIN, GENXBINS)
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

                d = d.Define("Muon_forTracking", f"Muon_isGlobal && Muon_pt > {minInnerTrackPt} && Muon_standalonePt > {minStandalonePt} && selfDeltaR(Muon_eta, Muon_phi, Muon_standaloneEta, Muon_standalonePhi) < 0.3 && Muon_highPurity")
                d = d.Define("Muon_standaloneNvalidHits", "getGlobalMuon_MergedStandAloneMuonVar(Muon_standaloneExtraIdx, MergedStandAloneMuon_extraIdx, MergedStandAloneMuon_numberOfValidHits)")
                d = d.Define("passHasHits",f"Muon_standaloneNvalidHits >= {minStandaloneNumberOfValidHits}")
                d = d.Define("Muon_forTrackingWithHits","Muon_forTracking && passHasHits")
                d = d.Define("Muon_forIsoNoTrig", "Muon_forTrackingWithHits && passCondition_IDIP && passCondition_Iso")
                d = d.Define("goodmuon","trackStandaloneDR(goodgeneta, goodgenphi, Muon_eta[Muon_forIsoNoTrig], Muon_phi[Muon_forIsoNoTrig]) < 0.3")
                d = d.Define("newweight","weight*goodmuon")
                pass_histogram_reco = d.Filter("goodgenpt.size()>=2").HistoND(model_pass_trig,strings_pass)
                pass_histogram_norm = d.Filter("goodgenpt.size()>=2").HistoND(model_norm_trig,strings_norm)

                ROOT.saveHistogramsGen(pass_histogram_reco,pass_histogram_norm,ROOT.std.string(args.output_file))

        else:
            if not (args.genLevelEfficiency):
                normFactor = args.normFactor
                scale = 1.0 if args.isData else (normFactor/weightSum.GetValue()) if args.normalizeMCsumGenWeights else normFactor
                makeAndSaveHistograms(d, histo_name, "IsolationNoTrigger", binning_mass, binning_pt, binning_eta, scaleFactor=scale)
            else:
                d = d.Define("Muon_forTracking", f"Muon_isGlobal && Muon_pt > {minInnerTrackPt} && Muon_standalonePt > {minStandalonePt} && selfDeltaR(Muon_eta, Muon_phi, Muon_standaloneEta, Muon_standalonePhi) < 0.3 && Muon_highPurity")
                d = d.Define("Muon_standaloneNvalidHits", "getGlobalMuon_MergedStandAloneMuonVar(Muon_standaloneExtraIdx, MergedStandAloneMuon_extraIdx, MergedStandAloneMuon_numberOfValidHits)")
                d = d.Define("passHasHits",f"Muon_standaloneNvalidHits >= {minStandaloneNumberOfValidHits}")
                d = d.Define("Muon_forTrackingWithHits","Muon_forTracking && passHasHits")
                d = d.Define("Muon_forIsoNoTrig", "Muon_forTrackingWithHits && passCondition_IDIP && passCondition_Iso")
                d = d.Define("goodmuon","trackStandaloneDR(goodgeneta, goodgenphi, Muon_eta[Muon_forIsoNoTrig], Muon_phi[Muon_forIsoNoTrig]) < 0.3")
                d = d.Define("newweight","weight*goodmuon")
                model_pass_reco = ROOT.RDF.TH2DModel("Pass","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)
                model_norm_reco = ROOT.RDF.TH2DModel("Norm","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)

                pass_histogram_reco = d.Histo2D(model_pass_reco,"goodgeneta","goodgenpt","newweight")
                pass_histogram_norm = d.Histo2D(model_norm_reco,"goodgeneta","goodgenpt","weight")

                pass_histogram_reco.Write()
                pass_histogram_norm.Write()

    ##For Isolation Failing Trigger

    if(args.efficiency == 7):

        # define condition for passing probes
        d = d.Define("passCondition_IDIPTrigIso", "passCondition_IDIP && !passCondition_Trig &&  passCondition_Iso")
        d = d.Define("failCondition_IDIPTrigIso", "passCondition_IDIP && !passCondition_Trig && !passCondition_Iso")
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
                model_pass_iso = ROOT.RDF.THnDModel("pass_mu_"+histo_name, "IsolationFailingTrigger_pass", 5, NBIN, XBINS)
                model_fail_iso = ROOT.RDF.THnDModel("fail_mu_"+histo_name, "IsolationFailingTrigger_fail", 5, NBIN, XBINS)
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
            if not (args.genLevelEfficiency):
                normFactor = args.normFactor
                scale = 1.0 if args.isData else (normFactor/weightSum.GetValue()) if args.normalizeMCsumGenWeights else normFactor
                makeAndSaveHistograms(d, histo_name, "Isolation", binning_mass, binning_pt, binning_eta, scaleFactor=scale)

    # For veto (loose ID+dxybs<0.05)
    if (args.efficiency == 8):
        if not (args.genLevelEfficiency):
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
            normFactor = args.normFactor
            scale = 1.0 if args.isData else (normFactor/weightSum.GetValue()) if args.normalizeMCsumGenWeights else normFactor
            makeAndSaveHistograms(d, histo_name, "newVeto", binning_mass, binning_pt, binning_eta, scaleFactor=scale)
        else:
            d = d.Define("Muon_forTracking", f"Muon_isGlobal && Muon_pt > {minInnerTrackPt} && Muon_standalonePt > {minStandalonePt} && selfDeltaR(Muon_eta, Muon_phi, Muon_standaloneEta, Muon_standalonePhi) < 0.3 && Muon_highPurity")
            d = d.Define("Muon_standaloneNvalidHits", "getGlobalMuon_MergedStandAloneMuonVar(Muon_standaloneExtraIdx, MergedStandAloneMuon_extraIdx, MergedStandAloneMuon_numberOfValidHits)")
            d = d.Define("passHasHits",f"Muon_standaloneNvalidHits >= {minStandaloneNumberOfValidHits}")
            d = d.Define("Muon_forTrackingWithHits","Muon_forTracking && passHasHits")
            d = d.Define("Muon_forVeto", "Muon_forTrackingWithHits && passCondition_veto")
            d = d.Define("goodmuon","trackStandaloneDR(goodgeneta, goodgenphi, Muon_eta[Muon_forVeto], Muon_phi[Muon_forVeto]) < 0.3")
            d = d.Define("newweight","weight*goodmuon")

            model_pass_reco = ROOT.RDF.TH2DModel("Pass","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)
            model_norm_reco = ROOT.RDF.TH2DModel("Norm","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)

            pass_histogram_reco = d.Histo2D(model_pass_reco,"goodgeneta","goodgenpt","newweight")
            pass_histogram_norm = d.Histo2D(model_norm_reco,"goodgeneta","goodgenpt","weight")

            pass_histogram_reco.Write()
            pass_histogram_norm.Write()

elif args.efficiency == 9:
    
    # P(tracker-seeded track | Standalone muon), which is the "tracking" efficiency for tracker-seeded track
    #
    # should try to reuse existing code defined above

    if(args.isData):
        d = d.Define("isGenMatchedMergedStandMuon","createTrues(nMergedStandAloneMuon)")
    else:
        d = d.Define("isGenMatchedMergedStandMuon","hasGenMatch(GenMuonBare_eta, GenMuonBare_phi, MergedStandAloneMuon_eta, MergedStandAloneMuon_phi, 0.3)")
        d = d.Define("GenMatchedIdx","GenMatchedIdx(GenMuonBare_eta, GenMuonBare_phi, MergedStandAloneMuon_eta, MergedStandAloneMuon_phi, 0.3)")

    # All probes, standalone muons from MergedStandAloneMuon_XX    
    d = d.Define("Probe_MergedStandMuons",f"MergedStandAloneMuon_pt > {minStandalonePt} && abs(MergedStandAloneMuon_eta) < 2.4 && isGenMatchedMergedStandMuon && MergedStandAloneMuon_numberOfValidHits >= {minStandaloneNumberOfValidHits}")
    #
    d = d.Define("All_TPPairs", f"CreateTPPair(Tag_Muons, Probe_MergedStandMuons, {doOStracking}, Tag_charge, MergedStandAloneMuon_charge, Tag_outExtraIdx, MergedStandAloneMuon_extraIdx, 0, {doSameCharge})")
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

    # define all probes
    d = d.Define("goodTracks", f"Track_pt > {minInnerTrackPt} && abs(Track_eta) < 2.4 && Track_trackOriginalAlgo != 13 && Track_trackOriginalAlgo != 14 && (Track_qualityMask & 4)")
    # note: the following function is symmetric between the two collections (even though the internally defined name seems to be asymmetric)
    # the important thing is just that the all–probes collctions is passed at the first two arguments
    d = d.Define("passCondition_tracking", "trackStandaloneDR(MergedStandAloneMuon_eta, MergedStandAloneMuon_phi, Track_eta[goodTracks], Track_phi[goodTracks]) < 0.3")

    default_pt_binning = [10., 15., 24., 35., 45., 55., 65.]
    default_pt_binning = [x for x in default_pt_binning if (x+0.1) >= minHistPt]
    binning_pt = array('d', default_pt_binning)

    d = d.Define("passCondition", "getVariables(TPPairs, passCondition_tracking, 2)")
    d = d.Define("failCondition", "!passCondition")

    # Here we are using the muon variables to calulate the mass for the passing probes for reco efficiency
    ## However the TPPairs were made using indices from MergedStandAloneMuon_XX collections, which are not necessarily valid for Muon_XX
    ## Thus, for each passing MergedStandAloneMuon I store the pt,eta,phi of the corresponding Muon (which exists as long as we use the MergedStandAloneMuon indices from TPPairs_pass)
    d = d.Define("TPPairs_pass", "TPPairs[passCondition]")
    d = d.Define("MergedStandaloneMuon_MatchedTrackIdx", "getMergedStandAloneMuon_highestPtTrackIdxWithinDR(MergedStandAloneMuon_eta, MergedStandAloneMuon_phi, Track_pt[goodTracks], Track_eta[goodTracks], Track_phi[goodTracks], 0.3)")
    d = d.Define("MergedStandaloneMuon_MatchedTrackPt",  "getMergedStandAloneMuon_matchedObjectVar(MergedStandaloneMuon_MatchedTrackIdx, Track_pt[goodTracks])")
    d = d.Define("MergedStandaloneMuon_MatchedTrackEta", "getMergedStandAloneMuon_matchedObjectVar(MergedStandaloneMuon_MatchedTrackIdx, Track_eta[goodTracks])")
    d = d.Define("MergedStandaloneMuon_MatchedTrackPhi", "getMergedStandAloneMuon_matchedObjectVar(MergedStandaloneMuon_MatchedTrackIdx, Track_phi[goodTracks])")

    d = d.Define("TPmass_pass",    "getTPmass(TPPairs_pass, Tag_pt, Tag_eta, Tag_phi, MergedStandaloneMuon_MatchedTrackPt, MergedStandaloneMuon_MatchedTrackEta, MergedStandaloneMuon_MatchedTrackPhi)")
    d = d.Define("Probe_pt_pass",  "Probe_pt[passCondition]")
    d = d.Define("Probe_eta_pass", "Probe_eta[passCondition]")

    d = d.Define("TPmass_fail",    "TPmass[failCondition]")
    d = d.Define("Probe_pt_fail",  "Probe_pt[failCondition]")
    d = d.Define("Probe_eta_fail", "Probe_eta[failCondition]")

    normFactor = args.normFactor
    scale = 1.0 if args.isData else (normFactor/weightSum.GetValue()) if args.normalizeMCsumGenWeights else normFactor

    makeAndSaveHistograms(d, histo_name, "Tracking_trackerSeededTrack", binning_mass, binning_pt, binning_eta, scaleFactor=scale)

    # save also the mass for passing probes computed with standalone variables
    # needed when making MC template for failing probes using all probes, since the mass should be consistently measured for both cases
    # do it also for data in case we want to check the difference in the efficiencies
    d = d.Define("TPmassFromSA_pass", "TPmass[passCondition]")
    makeAndSaveOneHist(d, f"{histo_name}_alt", "Tracking_trackerSeededTrack (mass from SA muons)",
                       binning_mass, binning_pt, binning_eta,
                       massVar="TPmassFromSA", isPass=True, scaleFactor=scale)

elif args.efficiency == 10:

    # P(tracker and not global muon | tracker-seeded track)
    # with tracker and global muons defined with all appropriate criteria 

    if(args.isData):
        d = d.Define("isGenMatchedTrack","createTrues(nTrack)")
    else:
        d = d.Define("isGenMatchedTrack", "hasGenMatch(  GenMuonBare_eta, GenMuonBare_phi, Track_eta, Track_phi)")
        d = d.Define("GenMatchedIdx",     "GenMatchedIdx(GenMuonBare_eta, GenMuonBare_phi, Track_eta, Track_phi)")

    chargeCut = ""
    if args.charge:
        sign= ">" if args.charge > 0 else "<"
        chargeCut = f" && Track_charge {sign} 0"

    # define all probes
    d = d.Define("Probe_Tracks", f"Track_pt > {minInnerTrackPt} && abs(Track_eta) < 2.4 && Track_trackOriginalAlgo != 13 && Track_trackOriginalAlgo != 14 && isGenMatchedTrack && (Track_qualityMask & 4) {chargeCut}")

    d = d.Define("All_TPPairs", f"CreateTPPair(Tag_Muons, Probe_Tracks, {doOS}, Tag_charge, Track_charge, Tag_inExtraIdx, Track_extraIdx, 0, {doSameCharge})")
    d = d.Define("All_TPmass", "getTPmass(All_TPPairs, Tag_pt, Tag_eta, Tag_phi, Track_pt, Track_eta, Track_phi)")
    d = d.Define("All_absDiffZ", "getTPabsDiffZ(All_TPPairs, Tag_Z, Track_Z)")

    # overriding previous pt binning
    default_pt_binning = [10., 15., 20., 24., 26., 30., 34., 38., 42., 46., 50., 55., 60., 65.]
    default_pt_binning = [x for x in default_pt_binning if (x+0.1) >= minHistPt]
    binning_pt = array('d', default_pt_binning)
    massLow  =  60
    massHigh = 120
    binning_mass = array('d',[massLow + i for i in range(int(1+massHigh-massLow))])
    massCut = f"All_TPmass > {massLow} && All_TPmass < {massHigh}"
    ZdiffCut = "All_absDiffZ < 0.2"
    d = d.Define("TPPairs", f"All_TPPairs[{massCut} && {ZdiffCut}]")
    d = d.Define("TPmass",  f"All_TPmass[{massCut}  && {ZdiffCut}]")

    d = d.Define("Probe_pt",   "getVariables(TPPairs, Track_pt,  2)")
    d = d.Define("Probe_eta",  "getVariables(TPPairs, Track_eta, 2)")

    if (args.tnpGenLevel):
        d = d.Redefine("Probe_pt","getGenVariables(TPPairs,GenMatchedIdx,GenMuonBare_pt,2)")
        d = d.Redefine("Probe_eta","getGenVariables(TPPairs,GenMatchedIdx,GenMuonBare_eta,2)")

    # condition for passing probe
    d = d.Define("Muon_standaloneNvalidHits", "getGlobalMuon_MergedStandAloneMuonVar(Muon_standaloneExtraIdx, MergedStandAloneMuon_extraIdx, MergedStandAloneMuon_numberOfValidHits)")
    d = d.Define("Muon_isGoodGlobal", f"Muon_isGlobal && Muon_standalonePt > {minStandalonePt} && selfDeltaR(Muon_eta, Muon_phi, Muon_standaloneEta, Muon_standalonePhi) < 0.3 && Muon_standaloneNvalidHits >= {minStandaloneNumberOfValidHits}")
    d = d.Define("Muon_isGoodTracker", "Muon_isTracker && Muon_innerTrackOriginalAlgo != 13 && Muon_innerTrackOriginalAlgo != 14")
    d = d.Define("Muon_forTracking", f"Muon_pt > {minInnerTrackPt} && Muon_highPurity && Muon_isGoodTracker && not Muon_isGoodGlobal")
    # 
    # check Muon exists with proper criteria and matching extraIdx with the tracker-seeded track 
    d = d.Define("passCondition_reco",
                 "Probe_isMatched(TPPairs, Track_extraIdx, Muon_innerTrackExtraIdx, Muon_forTracking)")
    d = d.Define("passCondition", "getVariables(TPPairs, passCondition_reco, 2)")
    d = d.Define("failCondition", "!passCondition")

    d = d.Define("Probe_pt_pass",  "Probe_pt[passCondition]")
    d = d.Define("Probe_eta_pass", "Probe_eta[passCondition]")
    d = d.Define("TPmass_pass", "TPmass[passCondition]")
    d = d.Define("Probe_pt_fail",  "Probe_pt[failCondition]")
    d = d.Define("Probe_eta_fail", "Probe_eta[failCondition]")
    d = d.Define("TPmass_fail", "TPmass[failCondition]")

    normFactor = args.normFactor
    scale = 1.0 if args.isData else (normFactor/weightSum.GetValue()) if args.normalizeMCsumGenWeights else normFactor
    makeAndSaveHistograms(d, histo_name, "Reco_trackerMuon", binning_mass, binning_pt, binning_eta, scaleFactor=scale)

elif args.efficiency == 11:

    # P(looseID + dxy_bs | tracker or global)

    if(args.isData != 1):
        d = d.Define("GenMatchedIdx","GenMatchedIdx(GenMuonBare_eta, GenMuonBare_phi, Muon_eta, Muon_phi)")

    chargeCut = ""
    if args.charge:
        sign= ">" if args.charge > 0 else "<"
        chargeCut = f" && Muon_charge {sign} 0"

    d = d.Define("Muon_standaloneNvalidHits", "getGlobalMuon_MergedStandAloneMuonVar(Muon_standaloneExtraIdx, MergedStandAloneMuon_extraIdx, MergedStandAloneMuon_numberOfValidHits)")
    d = d.Define("Muon_isGoodGlobal", f"Muon_isGlobal && Muon_standalonePt > {minStandalonePt} && selfDeltaR(Muon_eta, Muon_phi, Muon_standaloneEta, Muon_standalonePhi) < 0.3 && Muon_standaloneNvalidHits >= {minStandaloneNumberOfValidHits}")
    d = d.Define("Muon_isGoodTracker", "Muon_isTracker && Muon_innerTrackOriginalAlgo != 13 && Muon_innerTrackOriginalAlgo != 14")

    d = d.Define("BasicProbe_Muons", f"Muon_pt > {minInnerTrackPt} && abs(Muon_eta) < 2.4 && Muon_highPurity && isGenMatchedMuon {chargeCut} && (Muon_isGoodGlobal || Muon_isGoodTracker)")
    d = d.Define("All_TPPairs", f"CreateTPPair(Tag_Muons, BasicProbe_Muons, {doOS}, Tag_charge, Muon_charge, Tag_inExtraIdx, Muon_innerTrackExtraIdx, 1, {doSameCharge})") # these are all Muon_XX, so might just exclude same index in the loop
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

    if (args.tnpGenLevel):
        d = d.Redefine("BasicProbe_pt",  "getGenVariables(TPPairs,GenMatchedIdx,GenMuonBare_pt,2)")
        d = d.Redefine("BasicProbe_eta", "getGenVariables(TPPairs,GenMatchedIdx,GenMuonBare_eta,2)")

    d = d.Define("passCondition_veto", "Muon_looseId && abs(Muon_dxybs) < 0.05")

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

    if not args.genLevelEfficiency:
        normFactor = args.normFactor
        scale = 1.0 if args.isData else (normFactor/weightSum.GetValue()) if args.normalizeMCsumGenWeights else normFactor
        makeAndSaveHistograms(d, histo_name, "Veto", binning_mass, binning_pt, binning_eta, scaleFactor=scale)
    else:
        d = d.Define("Muon_forVeto", "Muon_pt > {minInnerTrackPt} && Muon_highPurity && (Muon_isGoodGlobal || Muon_isGoodTracker) && passCondition_veto")
        d = d.Define("goodmuon","trackStandaloneDR(goodgeneta, goodgenphi, Muon_eta[Muon_forVeto], Muon_phi[Muon_forVeto]) < 0.3")
        d = d.Define("newweight","weight*goodmuon")

        model_pass_reco = ROOT.RDF.TH2DModel("Pass","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)
        model_norm_reco = ROOT.RDF.TH2DModel("Norm","",len(binning_eta)-1,binning_eta,len(binning_pt)-1,binning_pt)

        pass_histogram_reco = d.Histo2D(model_pass_reco,"goodgeneta","goodgenpt","newweight")
        pass_histogram_norm = d.Histo2D(model_norm_reco,"goodgeneta","goodgenpt","weight")

        pass_histogram_reco.Write()
        pass_histogram_norm.Write()


#######
#######
#######
#######
#######

weightSumHist = ROOT.TH1D("weightSum","Sum of the sign of the gen weights",1,-0.5,0.5)
weightSumHist.SetBinContent(1, weightSum.GetValue())
weightSumHist.Write()
f_out.Close()

print(d.Report().Print())

elapsed = time.time() - tstart
elapsed_cpu = time.process_time() - cpustrat
print('Execution time:', elapsed, 'seconds')
print('CPU Execution time:', elapsed_cpu , 'seconds')
print()
