import ROOT
from array import array
import json

ROOT.gInterpreter.ProcessLine(".O3")
ROOT.ROOT.EnableImplicitMT()
ROOT.gInterpreter.Declare('#include "Steve.h"')
ROOT.gInterpreter.Declare('#include "GenFunctions.h"')
import os
#from os import listdir
import time
import sys

import argparse


cpp_code = """ 
    template <typename T>
    int debugFilter(RVec<T> var) {
        std::cout << var << std::endl;
        return 1;
        }
    """
ROOT.gInterpreter.ProcessLine(cpp_code)


def makeAndSaveOneHist(d, histo_name, histo_title, binning_mass, binning_pt, binning_eta, massVar="TPmass", isPass=True):

    passStr = "pass" if isPass else "fail"
    model = ROOT.RDF.TH3DModel(f"{passStr}_mu_{histo_name}", f"{histo_title} {passStr}",
                               len(binning_mass)-1, binning_mass,
                               len(binning_pt)-1, binning_pt,
                               len(binning_eta)-1, binning_eta)
    
    histogram = d.Histo3D(model, f"{massVar}_{passStr}", f"Probe_pt_{passStr}", f"Probe_eta_{passStr}", "weight")
    print(histogram.Integral())
    histogram.Write()
    
    
def makeAndSaveHistograms(d, histo_name, histo_title, binning_mass, binning_pt, binning_eta, massVar="TPmass"):

    # model_pass = ROOT.RDF.TH3DModel(f"pass_mu_{histo_name}", f"{histo_title} pass",
    #                                 len(binning_mass)-1, binning_mass,
    #                                 len(binning_pt)-1, binning_pt,
    #                                 len(binning_eta)-1, binning_eta)
    # model_fail = ROOT.RDF.TH3DModel(f"fail_mu_{histo_name}", f"{histo_title} fail",
    #                                 len(binning_mass)-1, binning_mass,
    #                                 len(binning_pt)-1, binning_pt,
    #                                 len(binning_eta)-1, binning_eta)
    
    # pass_histogram = d.Histo3D(model_pass, f"{massVar}_pass", "Probe_pt_pass", "Probe_eta_pass", "weight")
    # fail_histogram = d.Histo3D(model_fail, f"{massVar}_fail", "Probe_pt_fail", "Probe_eta_fail", "weight")
    
    # pass_histogram.Write()
    # fail_histogram.Write()

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
		    help="1 for reco, 2 for \"tracking\", 3 for idip, 4 for trigger, 5 for isolation, 6 for isolation without trigger, 7 for veto, 8 for isolation with failing trigger",
                    type=int, choices=range(1,9))

#parser.add_argument("-i","--input_path", help="path of the input root files", type=str)

parser.add_argument("-i","--input_path", nargs='+', default=[], help="path of the input root files")

parser.add_argument("-o","--output_file", help="name of the output root file",
                    type=str)

parser.add_argument("-d","--isData", help="Pass 0 for MC, 1 for Data, default is 0",
                    type=int, default=0)

parser.add_argument("-b","--isBkg", help="Pass 0 for MC and Data, 1 for Bkg, default is 0",
                    type=int, default=0)
parser.add_argument("-tpt","--tagPt", help="Minimum pt to select tag muons",
                    type=float, default=25.)

parser.add_argument("-tiso","--tagIso", help="Isolation threshold to select tag muons",
                    type=float, default=0.15)

parser.add_argument(        "--standaloneValidHits", help="Minimum number of valid hits for the standalone track (>= this value)",
                    type=int, default=1)

parser.add_argument("-c","--charge", help="Make efficiencies for a specific charge of the probe (-1/1 for positive negative, 0 for inclusive)",
                    type=int, default=0, choices=[-1, 0, 1])

parser.add_argument("-p","--eventParity", help="Select events with given parity for statistical tests, -1/1 for odd/even events, 0 for all (default)",
                    type=int, default=0, choices=[-1, 0, 1])

parser.add_argument('-nw', '--noVertexPileupWeight', action='store_true', help='Do not use weights for vertex z position')
#parser.add_argument("-vpw", "--vertexPileupWeight", action="store_true", help="Use weights for vertex z position versus pileup (only for MC)")

parser.add_argument("-nos", "--noOppositeCharge", action="store_true", help="Don't require opposite charges between tag and probe (including tracking, unless also using --noOppositeChargeTracking)")
parser.add_argument(        "--noOppositeChargeTracking", action="store_true", help="Don't require opposite charges between tag and probe for tracking")

parser.add_argument("-sc", "--SameCharge", action="store_true", help="Require the TP Pair to have same sign (fo bkg study)",
                     default=False)

parser.add_argument("-zqt","--zqtprojection", action="store_true", help="Efficiencies evaluated as a function of zqtprojection (only for trigger and isolation)")

parser.add_argument("-gen","--genLevelEfficiency", action="store_true", help="Compute MC truth efficiency")

parser.add_argument("-tpg","--tnpGenLevel", action="store_true", help="Compute tag-and-probe efficiencies for MC as a function of postVFP gen variables")

parser.add_argument("-y","--year", help="run year 2016, 2017, 2018",
                    type=str,default="2016")
parser.add_argument("-iso","--isoDefinition",help="Choose between the old and new isolation definition, 0 is old, 1 is new", default=1, choices = [0,1], type =int)

args = parser.parse_args()
tstart = time.time()
cpustrat = time.process_time()

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

if args.isData == 1:
    histo_name= "RunGtoH"
else:
    histo_name = "DY_postVFP"

files=[]

for i in range(len(args.input_path)):
    for root, dirnames, filenames in os.walk(args.input_path[i]):
        for filename in filenames:
            if '.root' in filename:
                try:
                    file_tmp = ROOT.TFile(os.path.join(root, filename), "READ")
                    if not file_tmp.IsZombie():
                        files.append(os.path.join(root, filename))
                    file_tmp.Close()
                except:
                    pass

print("Num files:  ", len(files))

if args.charge and args.efficiency in [2]:
    print("")
    print("   WARNING: charge splitting for tracking efficiency is implemented using the tag muon (with the other charge)")
    print("")


#print(files)
filenames = ROOT.std.vector('string')()

for name in files: filenames.push_back(name)

d = ROOT.RDataFrame("Events",filenames )


# had to hack, since definition of the vertex variables were not consistent throughout 
#various productions. Made alias of the new variables since the part where actual calculation 
#is done remains same
d = d.Alias("Muon_Z", "Muon_vz")
d = d.Alias("Muon_X", "Muon_vx")
d = d.Alias("Muon_Y", "Muon_vy")
d = d.Alias("Track_Z", "Track_vz")
d = d.Alias("Track_X", "Track_vx")
d = d.Alias("Track_Y", "Track_vy")

masshighut = 200

binning_pt = array('d',[24., 26., 28., 30., 32., 34., 36., 38., 40., 42., 44., 47., 50., 55., 60., 65.])
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

minStandaloneNumberOfValidHits = args.standaloneValidHits

if(args.isData == 1):
    d = d.Define("gen_weight","1")
else:
    d = d.Define("gen_weight", "clipGenWeight(genWeight)") #for now (testing)
weightSum = d.Sum("gen_weight")



print("Starting general cuts")
##General Cuts
if(args.year == "2016"):
    d = d.Filter("HLT_IsoMu24 || HLT_IsoTkMu24","HLT Cut")
else:
    d = d.Filter("HLT_IsoMu24","HLT Cut")

d = d.Filter("PV_npvsGood >= 1","NVtx Cut")

# for statistical tests (postfix to be added to file name)
if args.eventParity < 0:
    d = d.Filter("(event % 2) == 1") # odd
elif args.eventParity > 0:
    d = d.Filter("(event % 2) == 0") # even

doOS = 0 if args.noOppositeCharge else 1
doOStracking = 0 if args.noOppositeChargeTracking else doOS

SS = 1 if args.SameCharge else 0

if doOS & SS:
    raise Exception("Both doOS and SS can't be True. Require noOppositeCharge if you really want the Same Sign")

if doOStracking & SS:
    raise Exception("Both doOStracking and SS can't be True. Require noOppositeChargeTracking if you rally want Same Sign")

if (args.isData == 1):
    if (args.year == "2016"): 
        jsonhelper = make_jsonhelper("Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")
        d = d.Filter(jsonhelper,["run","luminosityBlock"],"jsonfilter")
    elif (args.year == "2018"):
        jsonhelper = make_jsonhelper("utility/Cert_314472-325175_13TeV_UL2018_Collisions18_HLT_IsoMu24_v_CustomJSON.txt")
        d = d.Filter(jsonhelper,["run","luminosityBlock"],"jsonfilter")
    elif (args.year == "2017"):
        print("2017  is not yet supported")
    else:
        print("Please provide correct year")


## Weights

if(args.isData == 1):
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
            else:
                print("Please provide a correct year")
                sys.exit(1)
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
            else:
                print("Please provide a correct year")
                sys.exit(1)
            ROOT.initializePileupWeights(input_PU_mc,input_PU_data)
            d = d.Define("pu_weight", "_get_PileupWeight(Pileup_nTrueInt,2)")
            #d = d.Define("pu_weight", "puw_2016(Pileup_nTrueInt,2)") # 2 is for postVFP

    d = d.Define("weight", "gen_weight*pu_weight*vertex_weight")
    

    
## For Tag Muons
if args.year == "2016":
    d = d.Define("isTriggeredMuon","hasTriggerMatch(Muon_eta, Muon_phi, TrigObj_id, TrigObj_filterBits, TrigObj_eta, TrigObj_phi)")
else:
    d =d.Define("isTriggeredMuon","hasTriggerMatch2018(Muon_eta, Muon_phi, TrigObj_id, TrigObj_filterBits, TrigObj_eta, TrigObj_phi)")

if(args.isData == 1):
    d = d.Define("isGenMatchedMuon","createTrues(nMuon)")
else: 
    d = d.Define("GenMuonBare", "GenPart_status == 1 && (GenPart_statusFlags & 1 || GenPart_statusFlags & (5<<1)) && abs(GenPart_pdgId) == 13")
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
if args.efficiency == 2 and args.charge:
    TagAntiChargeCut = " && Tag_charge < 0" if args.charge > 0 else " && Tag_charge > 0" # note that we swap charge 
# now define the tag muon (Muon_isGlobal might not be necessary, but shouldn't hurt really)
d = d.Define("Tag_Muons", f"Muon_pt > {args.tagPt} && abs(Muon_eta) < 2.4 && Muon_pfRelIso04_all < {args.tagIso} && abs(Muon_dxybs) < 0.05 && Muon_mediumId && Muon_isGlobal && isTriggeredMuon && isGenMatchedMuon {TagAntiChargeCut}")

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
f_out = ROOT.TFile(args.output_file, "RECREATE")
    
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
        # condition for passing probes
        # FIXME: add other criteria to the MergedStandAloneMuon to accept the matching? E.g. |eta| < 2.4 or pt > XX? No, it is an acceptance on numerator which we don't want
        d = d.Define("goodStandaloneMuon", f"MergedStandAloneMuon_pt > 15 && MergedStandAloneMuon_numberOfValidHits >= {minStandaloneNumberOfValidHits}")
        d = d.Define("passCondition_reco", "trackStandaloneDR(Track_eta, Track_phi, MergedStandAloneMuon_eta[goodStandaloneMuon], MergedStandAloneMuon_phi[goodStandaloneMuon]) < 0.3")
        
        d = d.Define("All_TPPairs", f"CreateTPPair(Tag_Muons, Probe_Tracks, {doOS}, Tag_charge, Track_charge, Tag_inExtraIdx, Track_extraIdx, 0, {SS})")
        d = d.Define("All_TPmass", "getTPmass(All_TPPairs, Tag_pt, Tag_eta, Tag_phi, Track_pt, Track_eta, Track_phi)")
        d = d.Define("All_absDiffZ", "getTPabsDiffZ(All_TPPairs, Tag_Z, Track_Z)")
        
        # overriding previous pt binning
        #binning_pt = array('d',[24., 65.])
        #binning_pt = array('d',[24., 26., 30., 34., 38., 42., 46., 50., 55., 65.])
        binning_pt = array('d',[24., 26., 30., 34., 38., 42., 46., 50., 55., 60., 65.])
        ## binning is currently 50,130 GeV, but it is overridden below 
        # also for mass
        massLow  =  60
        massHigh = 120
        binning_mass = array('d',[massLow + i for i in range(int(1+massHigh-massLow))])
        massCut = f"All_TPmass > {massLow} && All_TPmass < {massHigh}"
        ZdiffCut = "All_absDiffZ < 0.2"
        d = d.Define("TPPairs", f"All_TPPairs[{massCut} && {ZdiffCut}]")
        d = d.Define("TPmass",  f"All_TPmass[{massCut}  && {ZdiffCut}]")
        
        d = d.Define("Probe_pt",   "getVariables(TPPairs, Track_pt,  2)")
        d = d.Define("Probe_eta",  "getVariables(TPPairs, Track_eta, 2)")
        d = d.Define("passCondition", "getVariables(TPPairs, passCondition_reco, 2)")
        d = d.Define("failCondition", "!passCondition")

        if (args.tnpGenLevel):
            d = d.Redefine("Probe_pt","getGenVariables(TPPairs,GenMatchedIdx,GenMuonBare_pt,2)")
            d = d.Redefine("Probe_eta","getGenVariables(TPPairs,GenMatchedIdx,GenMuonBare_eta,2)")
        
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
        d = d.Define("Probe_MergedStandMuons",f"MergedStandAloneMuon_pt > 15 && abs(MergedStandAloneMuon_eta) < 2.4 && isGenMatchedMergedStandMuon && MergedStandAloneMuon_numberOfValidHits >= {minStandaloneNumberOfValidHits}")
        #
        d = d.Define("All_TPPairs", f"CreateTPPair(Tag_Muons, Probe_MergedStandMuons, {doOStracking}, Tag_charge, MergedStandAloneMuon_charge, Tag_outExtraIdx, MergedStandAloneMuon_extraIdx, 0, {SS})")
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

        # condition for passing probe, start from Muon_XX and then add match of extraID indices between Muon and MergedStandAloneMuon
        #d = d.Define("globalMuon_standaloneNvalidHits", "getGlobalMuon_MergedStandAloneMuonVar(Muon_standaloneExtraIdx, MergedStandAloneMuon_extraIdx, MergedStandAloneMuon_numberOfValidHits)")
        d = d.Define("Muon_forTracking", "Muon_isGlobal && Muon_pt > 10 && Muon_standalonePt > 15 && selfDeltaR(Muon_eta, Muon_phi, Muon_standaloneEta, Muon_standalonePhi) < 0.3 && Muon_highPurity")
        ## && globalMuon_standaloneNvalidHits >= {minStandaloneNumberOfValidHits}
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
        
        # use the minimum pt of the standalone muon used above to define the range, also a larger upper edge because the pt resolution of standalone muons is bad
        #binning_pt = array('d',[15., 80.]) # try also 24,65
        #binning_pt = array('d',[15., 30., 45., 60., 80.])
        #binning_pt = array('d',[15., 25., 35., 45., 55., 65., 80.])
        #binning_pt = array('d',[24., 65.])
        binning_pt = array('d',[24., 35., 45., 55., 65.])
        #binning_pt = array('d',[(15.0 + i*1.0) for i in range(66) ])
        
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
        
        print("Making histograms for tracking step")
        makeAndSaveHistograms(d, histo_name, "Tracking", binning_mass, binning_pt, binning_eta)

        # save also the mass for passing probes computed with standalone variables
        # needed when making MC template for failing probes using all probes, since the mass should be consistently measured for both cases
        # do it also for data in case we want to check the difference in the efficiencies
        d = d.Define("TPmassFromSA_pass", "TPmass[passCondition]")
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
        
    d = d.Define("globalMuon_standaloneNvalidHits", "getGlobalMuon_MergedStandAloneMuonVar(Muon_standaloneExtraIdx, MergedStandAloneMuon_extraIdx, MergedStandAloneMuon_numberOfValidHits)")
    d = d.Define("BasicProbe_Muons", f"Muon_isGlobal && Muon_pt > 24 && Muon_standalonePt > 15 && abs(Muon_eta) < 2.4 && selfDeltaR(Muon_eta, Muon_phi, Muon_standaloneEta, Muon_standalonePhi) < 0.3 && Muon_highPurity && isGenMatchedMuon {chargeCut} && globalMuon_standaloneNvalidHits >= {minStandaloneNumberOfValidHits}")
    # add MergedStandAloneMuon_numberOfValidHits > 0 matching the global muon to its standalone one 
    d = d.Define("All_TPPairs", f"CreateTPPair(Tag_Muons, BasicProbe_Muons, {doOS}, Tag_charge, Muon_charge, Tag_inExtraIdx, Muon_innerTrackExtraIdx, 1, {SS})") # these are all Muon_XX, so might just exclude same index in the loop
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

    ##For Isolation Failing Trigger

    if(args.efficiency == 8):

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

else:
    # for the veto selection
    # chargeCut = ""
    # if args.charge:
    #     sign= ">" if args.charge > 0 else "<"
    #     chargeCut = f" && Muon_charge {sign} 0"
    ## FIXME: add something else? Note that looseId already includes (Muon_isGlobal || Muon_isTracker)
    #d = d.Define("BasicProbe_Muons", f"Muon_pt > 10 && abs(Muon_eta) < 2.4 && (Muon_isGlobal || Muon_isTracker) && isGenMatchedMuon {chargeCut}")
    #
    # use tracks for all probes rather than muons
    if(args.isData == 1):
        d = d.Define("isGenMatchedTrack","createTrues(nTrack)")
    else:
        d = d.Define("isGenMatchedTrack", "hasGenMatch(  GenMuonBare_eta, GenMuonBare_phi, Track_eta, Track_phi)")
        d = d.Define("GenMatchedIdx",     "GenMatchedIdx(GenMuonBare_eta, GenMuonBare_phi, Track_eta, Track_phi)")
    #
    # FIXME: only tracker-seeded tracks here?
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

    # no need to require at least one valid hit for the standalone when the muon is global (I think)
    #d = d.Define("globalMuon_standaloneNvalidHits", "getGlobalMuon_MergedStandAloneMuonVar(Muon_standaloneExtraIdx, MergedStandAloneMuon_extraIdx, MergedStandAloneMuon_numberOfValidHits)")
    #d = d.Define("vetoMuons", f"Muon_pt > 10 && abs(Muon_eta) < 2.4 && ((Muon_isGlobal && globalMuon_standaloneNvalidHits >= {minStandaloneNumberOfValidHits}) || Muon_isTracker) && Muon_innerTrackOriginalAlgo != 13 && Muon_innerTrackOriginalAlgo != 14 && Muon_looseId && abs(Muon_dxybs) < 0.05 && Muon_highPurity")
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
    

weightSumHist = ROOT.TH1D("weightSum","Sum of the sign of the gen weights",1,-0.5,0.5)
weightSumHist.SetBinContent(1,weightSum.GetValue())
weightSumHist.Write()
        
f_out.Close()

print(d.Report().Print())

elapsed = time.time() - tstart
elapsed_cpu = time.process_time() - cpustrat
print('Execution time:', elapsed, 'seconds')
print('CPU Execution time:', elapsed_cpu , 'seconds')
