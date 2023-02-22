#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "TCanvas.h"
#include "TH1D.h"
#include "TLatex.h"
#include "Math/Vector4D.h"
#include "TStyle.h"
#include <string>

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string.hpp>

using namespace ROOT;
using namespace ROOT::VecOps;


/*class Probe{
  public:
  double pt;
  double eta;
  double phi;

  int charge;  
  int original_index;
  };*/

std::unordered_map<int, TH2D> hVertexPileupWeights = {}; // this has the scale factors as a function of vertex z and pileup to make the vertex distribution in MC agree with the one in data

void initializeVertexPileupWeights(const std::string& _filename_vertexPileupWeights = "./utility/vertexPileupWeights.root") {

  // weights vs vertex z and pileup
  TFile _file_vertexPileupWeights = TFile(_filename_vertexPileupWeights.c_str(), "read");
  if (!_file_vertexPileupWeights.IsOpen()) {
    std::cerr << "WARNING: Failed to open prefiring file " << _filename_vertexPileupWeights << "\n";
    exit(EXIT_FAILURE);
  }
  std::cout << "INFO >>> Initializing histograms for vertex-pileup weights from file " << _filename_vertexPileupWeights << std::endl;
  std::vector<std::string> eras = {"BtoF", "GtoH"};
  int id = 1;
  for (auto& era : eras) {
    std::vector<std::string> vars = {"weight_vertexZ_pileup", era};
    std::string corrname = boost::algorithm::join(vars, "_");
    auto* histptr = dynamic_cast<TH2D*>(_file_vertexPileupWeights.Get(corrname.c_str()));
    if (histptr == nullptr) {
        std::cerr << "WARNING: Failed to load correction " << corrname << " in file "
                  << _filename_vertexPileupWeights << "! Aborting" << std::endl;
        exit(EXIT_FAILURE);
    }
    histptr->SetDirectory(0);
    hVertexPileupWeights[id] = *dynamic_cast<TH2D*>(histptr);
    id++;
  }
  _file_vertexPileupWeights.Close();
  
}

double getValFromTH2(const TH2& h, const float& x, const float& y, const double& sumError=0.0) {
  //std::cout << "x,y --> " << x << "," << y << std::endl;
  int xbin = std::max(1, std::min(h.GetNbinsX(), h.GetXaxis()->FindFixBin(x)));
  int ybin  = std::max(1, std::min(h.GetNbinsY(), h.GetYaxis()->FindFixBin(y)));
  //std::cout << "xbin,ybin --> " << xbin << "," << ybin << std::endl;
  if (sumError)
    return h.GetBinContent(xbin, ybin) + sumError * h.GetBinError(xbin, ybin);
  else
    return h.GetBinContent(xbin, ybin);
}

double _get_vertexPileupWeight(const Float_t& vertexZ, const Float_t& nTrueInt, int era = 2) {
    // era = 1 for preVFP, 2 for postVFP
    // FIXME: for preVFP, histogram range for pileup is up to 45, but bin between 40 and 45 is empty, because there were few events with pileup > 40
    //        Rather than setting weights to 0 one should use the same as those from N-1 bin
    const TH2D& h = hVertexPileupWeights[era];
    return getValFromTH2(h, vertexZ, nTrueInt);
    
}

////=====================================================================================

template <typename T>
int printRvec(const RVec<T>  &vec, const int id = 0)
{
    std::cout << id << ": size = " << vec.size() << std::endl;
    return 1;
}

////=====================================================================================

float deltaPhi(float phi1, float phi2)
{                                                        
  float result = phi1 - phi2;
  while (result > float(M_PI)) result -= float(2*M_PI);
  while (result <= -float(M_PI)) result += float(2*M_PI);
  return result;
}

float deltaR2(float eta1, float phi1, float eta2, float phi2)
{
  float deta = std::abs(eta1-eta2);
  float dphi = deltaPhi(phi1,phi2);
  return deta*deta + dphi*dphi;
}

float deltaR(float eta1, float phi1, float eta2, float phi2)
{
  return std::sqrt(deltaR2(eta1,phi1,eta2,phi2));
}

RVec<Float_t> selfDeltaR(const RVec<Float_t>& eta1, const RVec<Float_t>& phi1, const RVec<Float_t>& eta2, const RVec<Float_t>& phi2, const float init = 999.9)
{
    RVec<Float_t> res(eta1.size(), init);
    for (unsigned int i = 0; i < res.size(); ++i) {    
        res[i] = std::sqrt(deltaR2(eta1[i], phi1[i] ,eta2[i] ,phi2[i]));
    }
    return res;
}


RVec<std::pair<int,int>> CreateTPPair(const RVec<Int_t> &Tag_muons, 
                                      const RVec<Int_t> &Probe_Candidates,
                                      const int doOppositeCharge,
                                      const RVec<Int_t> &Tag_Charge, 
                                      const RVec<Int_t> &Probe_charge,
                                      const RVec<Int_t> &Tag_extraIdx,
                                      const RVec<Int_t> &Probe_extraIdx,
                                      const int isSameCollection = false)
{

    // tag and probe collections might contain "same" physical objects, but here no DR cut is considered.
    // However, I imagine the mass cut would already reject these corner cases, but the cuts with extraIdx should take care of it here.
    // In addition, is isSameCollection is true we are then using the same collection for both tag and probe (e.g. could be Muon_XX),
    // so can just skip the loop entry with same indices
    
    RVec<std::pair<int,int>> TP_pairs;
    for (int iLep1 = 0; iLep1 < Tag_muons.size(); iLep1++) {
        if (not Tag_muons[iLep1]) continue;
        for(int iLep2 = 0; iLep2 < Probe_Candidates.size(); iLep2++){
            // Probe_Candidates is an RVec filled with 1 or 0 based on whether the elements from the initial collection
            // satisfy the probe selection (the one for denominator, i.e. all probes)
            // if probe_candidate = 0 we move on, otherwise we keep track of the position of the element 
            if (isSameCollection && (iLep1 == iLep2)) continue;
            if (not Probe_Candidates[iLep2]) continue;
            if (doOppositeCharge and (Tag_Charge[iLep1] == Probe_charge[iLep2])) continue;
            if (Tag_extraIdx[iLep1] == Probe_extraIdx[iLep2]) continue;
            std::pair<int,int> TP_pair = std::make_pair(iLep1, iLep2); 
            TP_pairs.push_back(TP_pair);
        }

    }
    // std::cout << "Found " << TP_pairs.size() << " tag-probe pairs" << std::endl;
    return TP_pairs;

}

RVec<Bool_t> hasStandAloneOrGlobalMatch(RVec<Float_t> &Track_eta, RVec<Float_t> &Track_phi,
					RVec<Float_t> &Muon_eta, RVec<Float_t> &Muon_phi,
					RVec<Bool_t> &Muon_isStandalone,
					RVec<Bool_t> &Muon_isGlobal)
{
  RVec<Int_t> hasStandAloneOrGlobalMatch;
  for(int iTrack=0;iTrack<Track_eta.size();iTrack++){
    bool has_match = false;
    for (int iMuon=0; iMuon<Muon_eta.size(); ++iMuon){
      if (!(Muon_isStandalone[iMuon] || Muon_isGlobal[iMuon])) continue;
      if (deltaR(Muon_eta[iMuon], Muon_phi[iMuon], Track_eta[iTrack], Track_phi[iTrack]) < 0.1)
      {
	has_match = 1; 
	break;
      }
    } 
    hasStandAloneOrGlobalMatch.push_back(has_match);
  }
  return hasStandAloneOrGlobalMatch;
}


RVec<Float_t> trackMuonDR(RVec<Float_t> &Track_eta, RVec<Float_t> &Track_phi,
			  RVec<Float_t> &Muon_eta,RVec<Float_t> &Muon_phi)
{
  RVec<Float_t> trackMuonDR; 
  for(int iTrack=0;iTrack<Track_eta.size();iTrack++){
    float dr = 999.;
    float tmp_dr  = 999.;

    for (unsigned int iMuon=0; iMuon<Muon_eta.size(); ++iMuon){
      tmp_dr  = deltaR(Muon_eta.at(iMuon), Muon_phi.at(iMuon), 
		       Track_eta.at(iTrack), Track_phi.at(iTrack));
      if (tmp_dr < dr) dr = tmp_dr;
    }
    trackMuonDR.push_back(dr);
  }  
  return trackMuonDR;
}


RVec<Float_t> trackStandaloneDR(const RVec<Float_t> &Track_eta, const RVec<Float_t> &Track_phi,
                                const RVec<Float_t> &Muon_standaloneEta, const RVec<Float_t> &Muon_standalonePhi)
{
   RVec<Float_t> trackStandaloneDR;
   for(int iTrack=0;iTrack<Track_eta.size();iTrack++){
     float dr = 999.;
     float tmp_dr  = 999.;
     
     for (unsigned int iMuon=0; iMuon<Muon_standaloneEta.size(); ++iMuon){
       tmp_dr  = deltaR(Muon_standaloneEta.at(iMuon), Muon_standalonePhi.at(iMuon), Track_eta.at(iTrack), Track_phi.at(iTrack));
       if (tmp_dr < dr) dr = tmp_dr;
     } 
     trackStandaloneDR.push_back(dr);
   }
   return trackStandaloneDR;
 }


RVec<Float_t> coll1coll2DR(const RVec<Float_t> &coll1_eta, const RVec<Float_t> &coll1_phi,
                           const RVec<Float_t> &coll2_eta, const RVec<Float_t> &coll2_phi)
{
    RVec<Float_t> resDR(coll1_eta.size(), 999.);
    for(int ic1 = 0; ic1 < coll1_eta.size(); ic1++){
        float dr = 999.;
        float tmp_dr  = 999.;
     
        for (unsigned int ic2 = 0; ic2 < coll2_eta.size(); ++ic2){
            tmp_dr  = deltaR(coll2_eta.at(ic2), coll2_phi.at(ic2), coll1_eta.at(ic1), coll1_phi.at(ic1));
            if (tmp_dr < dr) dr = tmp_dr;
        } 
        resDR[ic1] = dr;
    }
    return resDR;
}


RVec<Bool_t> hasTriggerMatch(RVec<Float_t> &Muon_eta, RVec<Float_t> &Muon_phi, 
                             RVec<Int_t> &TrigObj_id, RVec<Int_t> &TrigObj_filterBits,
                             RVec<Float_t> &TrigObj_eta, RVec<Float_t> &TrigObj_phi)
{
  RVec<Bool_t> TriggerMatch;
  for (int iMuon = 0; iMuon<Muon_eta.size(); iMuon++ ){
    bool hasTrigMatch = false;
    for (unsigned int iTrig=0; iTrig<TrigObj_id.size(); ++iTrig){
      if (TrigObj_id[iTrig]  != 13 ) continue;
      if (! (( TrigObj_filterBits[iTrig] & 16) || (TrigObj_filterBits[iTrig] & 32) ) ) continue;
      if (deltaR(Muon_eta[iMuon], Muon_phi[iMuon], TrigObj_eta[iTrig], TrigObj_phi[iTrig]) < 0.3) {
          hasTrigMatch = true;
          break;
      }
    }
    TriggerMatch.push_back(hasTrigMatch);
  }
  return TriggerMatch;
}

RVec<Bool_t> hasGenMatch(RVec<Int_t> &GenPart_pdgId, RVec<Int_t> &GenPart_status,
			 RVec<Int_t> &GenPart_statusFlags, RVec<Float_t> &GenPart_eta,
			 RVec<Float_t> &GenPart_phi, RVec<Float_t> &Cand_eta, 
			 RVec<Float_t> &Cand_phi, double dR_angle=0.1)
{
  RVec<Bool_t> isGenMatched;  
  for(int iCand=0;iCand<Cand_eta.size();iCand++){
    bool matchMC = false;   
    float mcmatch_tmp_dr = 999.;
    for(unsigned int iGen=0; iGen<GenPart_pdgId.size(); iGen++){
      if ( abs(GenPart_pdgId[iGen])==13 &&  GenPart_status[iGen] == 1 && 
	  (GenPart_statusFlags[iGen] & 1) ) {
	if (deltaR(Cand_eta[iCand], Cand_phi[iCand], 
	           GenPart_eta[iGen], GenPart_phi[iGen]) < mcmatch_tmp_dr){
        mcmatch_tmp_dr  = deltaR(Cand_eta[iCand], Cand_phi[iCand], 
                                 GenPart_eta[iGen], GenPart_phi[iGen]);
	  //truePt     = GenPart_pt[ii];
	  //trueEta    = GenPart_eta[ii];
	  //trueCharge = GenPart_charge[ii];
	} 
      }
    }
    if (mcmatch_tmp_dr < dR_angle) matchMC = true;
    isGenMatched.push_back(matchMC);
  }
  return isGenMatched;
}


RVec<Int_t> GenMatchedIdx(RVec<Int_t> &GenPart_pdgId, RVec<Int_t> &GenPart_status,RVec<Int_t> &GenPart_statusFlags,RVec<Float_t> &GenPart_eta,RVec<Float_t> &GenPart_phi,RVec<Float_t> &Cand_eta, RVec<Float_t> &Cand_phi, const float coneDR = 0.1){
  RVec<Int_t> isGenMatched;  
  for(int iCand=0;iCand<Cand_eta.size();iCand++){
    int matchIdx = -1;   
    float mcmatch_tmp_dr = 999.;
    for(unsigned int iGen=0; iGen<GenPart_pdgId.size(); iGen++){
      if ( abs(GenPart_pdgId[iGen])==13 &&  GenPart_status[iGen] == 1 && (GenPart_statusFlags[iGen] & 1)  ) {
	if (deltaR(Cand_eta[iCand], Cand_phi[iCand], GenPart_eta[iGen], GenPart_phi[iGen]) < mcmatch_tmp_dr){
	  mcmatch_tmp_dr  = deltaR(Cand_eta[iCand], Cand_phi[iCand], GenPart_eta[iGen], GenPart_phi[iGen]);
      matchIdx=iGen;
	  //truePt     = GenPart_pt[ii];
	  //trueEta    = GenPart_eta[ii];
	  //trueCharge = GenPart_charge[ii];
	} 
      }
    }
    if (mcmatch_tmp_dr < coneDR) isGenMatched.push_back(matchIdx);
    else isGenMatched.push_back(-1);
  }
  return isGenMatched;
}


// use directly the collection of gen muons define in the apporpriate way
RVec<Bool_t> hasGenMatch(RVec<Float_t> &Gen_eta, RVec<Float_t> &Gen_phi,
                         RVec<Float_t> &Cand_eta, RVec<Float_t> &Cand_phi,
                         const float coneDR = 0.1)
{
  RVec<Bool_t> isGenMatched;  
  float tmpDR = 0.0; // utility variable filled below everytime
  for(int iCand=0;iCand<Cand_eta.size();iCand++){
    bool matchMC = false;   
    float mcmatch_tmp_dr = 999.;
    for(unsigned int iGen=0; iGen<Gen_eta.size(); iGen++){
        tmpDR = deltaR(Cand_eta[iCand], Cand_phi[iCand], Gen_eta[iGen], Gen_phi[iGen]);
        if (tmpDR < mcmatch_tmp_dr){
            mcmatch_tmp_dr = tmpDR;
        } 
    }
    if (mcmatch_tmp_dr < coneDR) matchMC = true;
    isGenMatched.push_back(matchMC);
  }
  return isGenMatched;
}


RVec<Int_t> GenMatchedIdx(RVec<Float_t> &GenPart_eta,RVec<Float_t> &GenPart_phi,RVec<Float_t> &Cand_eta, RVec<Float_t> &Cand_phi, const float coneDR = 0.1){

    RVec<Int_t> isGenMatched(Cand_eta.size(), -1); //  fill with proper size and initialize to -1  
    float tmpDR = 0.0; // utility variable filled below everytime
    for(int iCand=0;iCand<Cand_eta.size();iCand++){
        int matchIdx = -1;   
        float mcmatch_tmp_dr = 999.;
        for(unsigned int iGen=0; iGen<GenPart_eta.size(); iGen++){
            tmpDR = deltaR(Cand_eta[iCand], Cand_phi[iCand], GenPart_eta[iGen], GenPart_phi[iGen]);
            if (tmpDR < mcmatch_tmp_dr){
                mcmatch_tmp_dr = tmpDR;
                matchIdx=iGen;
            } 
        }
        if (mcmatch_tmp_dr < coneDR) isGenMatched[iCand] = matchIdx;
        else isGenMatched[iCand] = -1;
    }
    return isGenMatched;
}

RVec<Float_t> getTPmass(RVec<std::pair<int,int>> TPPairs,
                        RVec<Float_t> &Muon_pt, RVec<Float_t> &Muon_eta, RVec<Float_t> &Muon_phi, 
                        RVec<Float_t> &Cand_pt, RVec<Float_t> &Cand_eta, RVec<Float_t> &Cand_phi)
{
    // muon mass is used below
    RVec<Float_t> TPMass;
    for (int i=0;i<TPPairs.size();i++){
        std::pair<int,int> TPPair = TPPairs.at(i);
        int tag_index = TPPair.first;
        int probe_index = TPPair.second;
        ROOT::Math::PtEtaPhiMVector tag(  Muon_pt[tag_index],   Muon_eta[tag_index],   Muon_phi[tag_index],   0.106);
        ROOT::Math::PtEtaPhiMVector probe(Cand_pt[probe_index], Cand_eta[probe_index], Cand_phi[probe_index], 0.106);
        TPMass.push_back( (tag + probe).mass() );
    }
    return TPMass;
    
}

RVec<Float_t> getTPabsDiffZ(RVec<std::pair<int,int>> TPPairs,
                            RVec<Float_t> &tag_Z, RVec<Float_t> &probe_Z)
{
    RVec<Float_t> TPAbsDiffZ;
    for (int i = 0; i < TPPairs.size(); i++){
        std::pair<int,int> TPPair = TPPairs.at(i);
        int tag_index = TPPair.first;
        int probe_index = TPPair.second;        
        TPAbsDiffZ.push_back( std::fabs(tag_Z[tag_index] - probe_Z[probe_index]) );
    }
    return TPAbsDiffZ;
    
}

template <typename T>
RVec<T> getVariables(RVec<std::pair<int,int>> TPPairs,
                     RVec<T>  &Cand_variable, 
                     int option /*1 for tag and 2 for probe*/)
{
    RVec<T>  Variables(TPPairs.size(), 0);
    for (int i = 0; i < TPPairs.size(); i++){
        std::pair<int, int> TPPair = TPPairs.at(i);
        T variable; 
        if (option==1)      variable = Cand_variable.at(TPPair.first);
        else if (option==2) variable = Cand_variable.at(TPPair.second);
        Variables[i] = variable;
    }
    // if (TPPairs.size()) {
    //     std::cout << "in getVariables(), size = " << Variables.size() << std::endl;
    // }
    return Variables;
}


RVec<Float_t> getGenVariables(RVec<std::pair<int,int>> TPPairs, RVec<int> &GenMatchedIdx, RVec<Float_t> &Cand_variable, int option /*1 for tag and 2 for probe*/){
  RVec<Float_t> Variables;
  for (int i=0;i<TPPairs.size();i++){
    std::pair<int,int> TPPair = TPPairs.at(i);
    float variable; 
    if (option==1) variable = Cand_variable.at(GenMatchedIdx.at(TPPair.first));
    else if (option==2) variable = Cand_variable.at(GenMatchedIdx.at(TPPair.second));
    Variables.push_back(variable);
  }
  return Variables;
}

RVec<Float_t> zqtprojection(RVec<std::pair<int,int>> &TPPairs, RVec<Float_t> &Muon_pt, 
			    RVec<Float_t> &Muon_eta, RVec<Float_t> &Muon_phi) 
{
  RVec<Float_t> v;
  for (int i=0;i<TPPairs.size();i++){
    std::pair<int,int> TPPair = TPPairs.at(i);
    TLorentzVector tag, probe;
    tag.SetPtEtaPhiM(Muon_pt[TPPair.first], Muon_eta[TPPair.first], Muon_phi[TPPair.first], 0.);
    probe.SetPtEtaPhiM(Muon_pt[TPPair.second], Muon_eta[TPPair.second], 
		       Muon_phi[TPPair.second], 0.);
    TVector3 Tag(tag.Px(),tag.Py(),0.), Probe(probe.Px(), probe.Py(), 0.);
    v.emplace_back((Tag+Probe).Dot(Probe)/sqrt(Probe.Dot(Probe)));
  }
  return v;
}

RVec<Float_t> zqtprojectionGen(RVec<std::pair<int,int>> &TPPairs, RVec<int> &GenMatchedIdx, RVec<Float_t> &GenPart_pt, RVec<Float_t> &GenPart_eta, RVec<Float_t> &GenPart_phi) {
  RVec<Float_t> v;
  for (int i=0;i<TPPairs.size();i++){
    std::pair<int,int> TPPair = TPPairs.at(i);
    TLorentzVector tag, probe;
    tag.SetPtEtaPhiM(GenPart_pt[GenMatchedIdx[TPPair.first]],GenPart_eta[GenMatchedIdx[TPPair.first]],GenPart_phi[GenMatchedIdx[TPPair.first]],0.);
    probe.SetPtEtaPhiM(GenPart_pt[GenMatchedIdx[TPPair.second]],GenPart_eta[GenMatchedIdx[TPPair.second]],GenPart_phi[GenMatchedIdx[TPPair.second]],0.);
    TVector3 Tag(tag.Px(),tag.Py(),0.), Probe(probe.Px(), probe.Py(), 0.);
    v.emplace_back((Tag+Probe).Dot(Probe)/sqrt(Probe.Dot(Probe)));
  }
  return v;
}

RVec<Bool_t> isOS(RVec<std::pair<int,int>> TPPairs, RVec<Int_t> Muon_charge, 
		  RVec<Int_t> Cand_charge)
{
  RVec<Bool_t> isOS;
  for (int i=0;i<TPPairs.size();i++){
    std::pair<int,int> TPPair = TPPairs.at(i);
    int tag_index = TPPair.first;
    int probe_index = TPPair.second;
    bool os = false;
    if (Muon_charge[tag_index] != Cand_charge[probe_index]) os = true; //initially ==. We want this to return true for opposite sign pairs, right?
    isOS.push_back(os);
  }
  return isOS;
}


RVec<Int_t> Probe_isMatched(const RVec<std::pair<int,int>> &TPPairs, 
                            const RVec<Int_t> &probe_extraIdx, 
                            const RVec<Int_t> &target_extraIdx, 
                            const RVec<Int_t> &target_passProbeCondition)
{
    //RVec<Int_t> res(TPPairs.size(), 0);
    RVec<Int_t> res(probe_extraIdx.size(), 0);
    // std::cout << "TPPairs.size() = " << TPPairs.size() << std::endl;
    // std::cout << "probe_extraIdx.size() = " << probe_extraIdx.size() << std::endl;
    // std::cout << "target_extraIdx.size() = " << target_extraIdx.size()  << std::endl;
    // std::cout << "target_passProbeCondition.size() = " << target_passProbeCondition.size()  << std::endl;
    for (unsigned int i = 0; i < TPPairs.size(); i++) {
        int probeId = TPPairs.at(i).second;
        for (unsigned int j = 0; j < target_extraIdx.size(); j++) {
            // std::cout << "Probe id = " << TPPairs.at(i).second << std::endl;
            if ( (probe_extraIdx[probeId] == target_extraIdx[j]) && target_passProbeCondition[j] ) {
                //if ( probe_extraIdx[TPPairs.at(i).second] == target_extraIdx[j]) {
                // res[i] = 1;
                res[probeId] = 1;
                // std::cout << "Target match with index = " << j << std::endl;
                // std::cout << "Target extraIdx = " << target_extraIdx[j] << std::endl;
                // std::cout << "Probe extraIdx = " << probe_extraIdx[TPPairs.at(i).second] << std::endl;
                break;
            }
        }
    }
    // printRvec(res, 1);
    // for (unsigned int i = 0; i < res.size(); i++) {
    //     std::cout << i << "=" << res[i] << "   ";
    // }
    // std::cout << " " << std::endl;
    return res;
}

// particular case of previous function, could be exchanged for that
// actually this function is probably wrong since the returned size is not the one of the MergedStandAloneMuon_extraIdx collection
RVec<Int_t> Probe_isGlobal(const RVec<std::pair<int,int>> &TPPairs, 
                           const RVec<Int_t> &MergedStandAloneMuon_extraIdx, 
                           const RVec<Int_t> &Muon_standaloneExtraIdx, 
                           const RVec<Int_t> &Muon_passProbeCondition)
{
    RVec<Int_t> isGlobal(TPPairs.size(), 0);
    for (unsigned int i = 0; i < TPPairs.size(); i++) {
        for (unsigned int j = 0; j < Muon_standaloneExtraIdx.size(); j++) {
            if ( (MergedStandAloneMuon_extraIdx[TPPairs.at(i).second] == Muon_standaloneExtraIdx[j]) && Muon_passProbeCondition[j]) {
                isGlobal[i] = 1;
                break;
            }
        }
    }
    return isGlobal;
}

// this one was for tests, checking again the inner track extraId between tag and probe when the probe is a global muon.
// If the outer track where already checked and not overlapping between tag and probe, then it was found that the inner tracks are never the same
RVec<Int_t> Probe_isGlobal_checkExtraIdxTagInnerTrack(const RVec<std::pair<int,int>> &TPPairs, 
                                                      const RVec<Int_t> &MergedStandAloneMuon_extraIdx, 
                                                      const RVec<Int_t> &Muon_standaloneExtraIdx, 
                                                      const RVec<Int_t> &Muon_passProbeCondition,
                                                      const RVec<Int_t> &Tag_innerTrackExtraIdx,
                                                      const RVec<Int_t> &Muon_innerTrackExtraIdx)
{
    RVec<Int_t> isGlobal(TPPairs.size(), 0);
    for (auto i=0U; i<TPPairs.size(); i++) {
        std::pair<int,int> TPPair = TPPairs.at(i);
        int tag_index = TPPair.first;
        int probe_index = TPPair.second;
        int condition = 0;
        for (unsigned int j = 0; j < Muon_standaloneExtraIdx.size(); j++) {
            if ( (MergedStandAloneMuon_extraIdx[probe_index] == Muon_standaloneExtraIdx[j]) && Muon_passProbeCondition[j]) {
                // only accept cases where the matched global muon does not coincide with the tag 
                if (Tag_innerTrackExtraIdx[tag_index] != Muon_innerTrackExtraIdx[j]) {
                    condition = 1;
                }
                break;
            }
        }
        isGlobal[i] = condition;
    }
    return isGlobal;
}


RVec<Int_t> getMergedStandAloneMuon_MuonIdx(const RVec<Int_t> &MergedStandAloneMuon_extraIdx, 
                                            const RVec<Int_t> &Muon_standaloneExtraIdx)
{
    RVec<Int_t> res(MergedStandAloneMuon_extraIdx.size(), -1); // initialize to invalid index
    for (unsigned int i = 0; i < MergedStandAloneMuon_extraIdx.size(); i++) {
        for (unsigned int j = 0; j < Muon_standaloneExtraIdx.size(); j++) {
            if ( MergedStandAloneMuon_extraIdx[i] == Muon_standaloneExtraIdx[j] ) {
                res[i] = j;
            }
        }
    }
    return res;

}


// given SA muon get idx of matched track within DR and if more are found pick the one with highest pt
RVec<Int_t> getMergedStandAloneMuon_highestPtTrackIdxWithinDR(const RVec<Float_t> &sa_eta, const RVec<Float_t> &sa_phi,
                                                                const RVec<Float_t> &track_pt, const RVec<Float_t> &track_eta, const RVec<Float_t> &track_phi,
                                                                const Float_t coneDR = 0.3)
{
    RVec<Int_t> resIdx(sa_eta.size(), -1.);

    for(unsigned int isa = 0; isa < sa_eta.size(); isa++){

        float ptmax  = 0.0; // to search track with highest pt, as a start any track within DR will be accepted
        int idx = -1;
        
        for (unsigned int it = 0; it < track_eta.size(); ++it){
            if (deltaR(track_eta[it], track_phi[it], sa_eta[isa], sa_phi[isa]) < coneDR) {
                if (track_pt[it] > ptmax) {
                    ptmax = track_pt[it];
                    resIdx[isa] = it;
                }
            }
        }
    }
    
    return resIdx;

}

template <typename T>
RVec<T> getMergedStandAloneMuon_matchedObjectVar(const RVec<Int_t> &MergedStandAloneMuon_matchedObjIdx, 
                                                 const RVec<T> &matchedObj_var,
                                                 const T invalidValue = -99)
{
    // MergedStandAloneMuon_matchedObjIdx can be created using getMergedStandAloneMuon_MuonIdx if the target collection is Muon,
    // or getMergedStandAloneMuon_highestPtTrackIdxWithinDR if it is a track
    // the assumption is that this function will be used only for MergedStandAloneMuon elements for which the corresponding matched object was already checked to exist
    RVec<T> res(MergedStandAloneMuon_matchedObjIdx.size(), invalidValue); // initialize to default value for invalid indices
    int index = -1;
    for (unsigned int i = 0; i < MergedStandAloneMuon_matchedObjIdx.size(); i++) {
        index = MergedStandAloneMuon_matchedObjIdx[i];
        if (index >= 0) res[i] = matchedObj_var[index];
    }
    return res;

}

// return a collection equipollent to global muons, containing the variable for the associated standalone track
// similar to what getMergedStandAloneMuon_matchedObjectVar does for standalone muons
template <typename T>
RVec<T> getGlobalMuon_MergedStandAloneMuonVar(const RVec<Int_t> &Muon_standaloneExtraIdx,
                                              const RVec<Int_t> &MergedStandAloneMuon_extraIdx,
                                              const RVec<T> &MergedStandAloneMuon_variable)
{
    RVec<T> res(Muon_standaloneExtraIdx.size(), -1); // initialize to invalid index
    for (unsigned int ig = 0; ig < Muon_standaloneExtraIdx.size(); ig++) {
        for (unsigned int isa = 0; isa < MergedStandAloneMuon_extraIdx.size(); isa++) {
            if ( Muon_standaloneExtraIdx[ig] == MergedStandAloneMuon_extraIdx[isa]) {
                res[ig] = MergedStandAloneMuon_variable[isa];
            }
        }
    }
    return res;

}


float clipGenWeight(float genWeight)
{
  //float sign =  genWeight/ abs(genWeight); // these ratios are nasty sometimes
  float sign = std::copysign(1., genWeight); // return 1 with the sign of genWeight
  return sign;
}


RVec<Bool_t> createTrues(int size)
{
   RVec<Bool_t> Trues(size,true);
   return Trues;
}


// define PU weights as global static variable
static double _pileupWeights_2016UL_preVFP[100] = {0.2504131118100572, 0.3939321013157489, 0.8526636158826547, 0.9866213669526419, 1.190215852465404, 1.1717732155849212, 1.326891895326891, 1.302128523414722, 1.2168725803758078, 1.1749255163178758, 1.1447644053358441, 1.102713304466158, 1.0692501779890462, 1.0550975672887584, 1.0569511994368632, 1.0729183940184346, 1.0872387106666015, 1.1011786998112356, 1.1056898819281813, 1.101303498400071, 1.088286672084464, 1.0718292808747956, 1.058079574499241, 1.0449648261552615, 1.025695752731563, 1.00046910455114, 0.965609276269846, 0.923878093642559, 0.8813681476454888, 0.8400510076526477, 0.798328868347709, 0.761260571593272, 0.7314034121263029, 0.7061304389022356, 0.6819866841190891, 0.6603991169581992, 0.6410919007298677, 0.6223846217721132, 0.605687078856489, 0.5930840812694836, 0.582011303650068, 0.5781708392917653, 0.5827659365926185, 0.5992578284204035, 0.6356323085478808, 0.6626791743184136, 0.7155301587041382, 0.751988682685627, 0.7981531968150425, 0.8050985639827735, 0.7868927507088908, 0.7454549444968995, 0.6465468874666872, 0.5364058298875525, 0.47499187516544383, 0.3482692962400063, 0.2632917619857064, 0.17751675327388303, 0.1377624652170653, 0.1149203417849365, 0.08616115852929608, 0.06222937238005178, 0.042695760425515435, 0.04004166808477174, 0.01974930104922166, 0.01647184503725847, 0.06290628342997077, 0.01303886642827827, 0.013667854369381956, 1.0, 1.0, 0.0019529063925951538, 0.0032365977926854385, 1.0, 1.0, 0.0005276241856562116, 1.0, 1.0, 1.0, 0.00016103753557202973, 1.0388527323106208e-05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

static double _pileupWeights_2016UL_postVFP[100] = {0.2779610550399915, 0.35168574986879175, 0.9401358131220096, 0.7172607749435421, 0.6450278243068622, 0.34514283732429957, 0.1611589141974616, 0.11228086636773361, 0.1128907001750219, 0.132248161931188, 0.23663365123479801, 0.3748294200454673, 0.49634027107576173, 0.5788690583768407, 0.6337386853819873, 0.6829452947698369, 0.7290431580271535, 0.771240232782898, 0.8053320518674009, 0.838145513094554, 0.8732137242554343, 0.9141744181800968, 0.9620176337543163, 1.0117461834726504, 1.0592899699789526, 1.1108798663176178, 1.1665249980457664, 1.2278912372416946, 1.2978032479847579, 1.3734228274583853, 1.4464864083532345, 1.5217674744443201, 1.6042012731236663, 1.6903024407379545, 1.7736002585270576, 1.8587194366394921, 1.945388713886392, 2.0267678295124067, 2.103193805887598, 2.176901779641042, 2.2325068336824807, 2.2853919797700124, 2.3352157415420938, 2.3902044532666786, 2.474527728196408, 2.4672726014266892, 2.496437077366503, 2.4109004529454627, 2.3100068766977055, 2.0729403043523136, 1.7851603570781098, 1.4868763727687704, 1.1436607286230618, 0.8611002919532451, 0.720199723366783, 0.5270297520641577, 0.4232092942842518, 0.3207745497560822, 0.2913258236130013, 0.28994313455300363, 0.2595939673337032, 0.2213468961151929, 0.17617832286578322, 0.18811385931872351, 0.10383430153310313, 0.09553214426384232, 0.39783568721356927, 0.08910667460692347, 0.10021460424296642, 1.0, 1.0, 0.01716284582706522, 0.030004459306439593, 1.0, 1.0, 0.005676828768647521, 1.0, 1.0, 1.0, 0.0021085121280920915, 0.0001433866510502697, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

static double _pileupWeights_2016UL_all[100] = {0.2632356557554427, 0.3742679945096482, 0.8933786734001985, 0.8612440202897551, 0.936451083487855, 0.7870074904870388, 0.7842865089890744, 0.7482986514919006, 0.7030100369691004, 0.6895979517202786, 0.7220632836803534, 0.763910403022603, 0.8025819045425514, 0.8334308921185805, 0.8599614999010246, 0.891400406962681, 0.920511990765391, 0.9476046032834685, 0.9658844727496724, 0.9788132346809981, 0.9881782065639063, 0.99844680043953, 1.013366310557985, 1.0295027823268927, 1.0413326125642286, 1.0518612115035215, 1.0591280791177433, 1.0653849154487818, 1.0752032129322615, 1.0883157707092646, 1.1000221189424457, 1.1152482759080458, 1.1376583849438306, 1.1642259358328428, 1.1900922407614898, 1.2181726985095205, 1.2481936021128097, 1.276072820817388, 1.302720815842171, 1.3302923682657466, 1.3502556436654813, 1.3728188431559898, 1.398466203158693, 1.4328769341812313, 1.491569796604876, 1.5026503594434713, 1.5444761542923757, 1.5241504697957315, 1.5018649072481693, 1.3952317835832393, 1.2515492277844895, 1.0905590695743195, 0.8779349087382351, 0.6875390369622981, 0.5891270170548337, 0.4314756460341123, 0.3377274310451597, 0.24419793435431333, 0.20924050243363612, 0.19638694831074852, 0.1668876866007459, 0.1362926671800032, 0.10482693328817545, 0.1089637710285652, 0.05888774433883642, 0.05327147659976019, 0.21880347556380075, 0.048445604603131555, 0.05395215052606778, 1.0, 1.0, 0.009032568027473709, 0.015696042685489516, 1.0, 1.0, 0.0029243875845773007, 1.0, 1.0, 1.0, 0.001067514594009139, 7.229421196561938e-05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};


double puw_2016(float nTrueInt, int period=0)
{
    // NanoAOD store nTrueInt as float even though it should be an integer
    // cast it back to int below
    if (nTrueInt<100) {
        if      (period == 0) return _pileupWeights_2016UL_all[static_cast<int>(nTrueInt)];
        else if (period == 1) return _pileupWeights_2016UL_preVFP[static_cast<int>(nTrueInt)];
        else if (period == 2) return _pileupWeights_2016UL_postVFP[static_cast<int>(nTrueInt)];
        else {
            std::cout<< " you are giving a wrong period to the pileup weight function, returning 0 for the events!!!!" << std::endl;
            return 0.;
        }
    }
    else  return 1.;
}


class JsonHelper {
public:
  using pair_t = std::pair<unsigned int, unsigned int>;
  using jsonmap_t = std::unordered_map<unsigned int, std::vector<pair_t>>;
  
  JsonHelper(const std::vector<unsigned int> &runs, const std::vector<unsigned int> &firstlumis, 
	     const std::vector<unsigned int> &lastlumis) :
  jsonmap_(std::make_shared<jsonmap_t>()) {
    for (unsigned int i = 0; i < firstlumis.size(); ++i) {
      (*jsonmap_)[runs[i]].push_back(std::make_pair(firstlumis[i],lastlumis[i]));
    }
    
    for (auto &item : *jsonmap_) {
      std::sort(item.second.begin(), item.second.end());
    }
  }
  
  bool operator () (unsigned int run, unsigned int lumi) const {
    if (run == 1) {
      return true;
    }
    
    const auto it = jsonmap_->find(run);
    if (it != jsonmap_->end()) {
      auto const &pairs = it->second;
      auto const pairit = std::lower_bound(pairs.begin(), pairs.end(), lumi, 
			[](const pair_t &pair, unsigned int val) { return pair.second < val; } );
      if (pairit != pairs.end()) {
        if (lumi >= pairit->first) {
          return true;
        }
      }
    }
    return false;
  }
  

private:
  std::shared_ptr<jsonmap_t> jsonmap_;
};

void saveHistograms(ROOT::RDF::RResultPtr<THnT<double> > histo_pass, 
		    ROOT::RDF::RResultPtr<THnT<double> > histo_fail, 
                    std::string output_file) 
{
  THnD Histo_pass=*(THnD*)histo_pass.GetPtr()->Clone();
  THnD Histo_fail=*(THnD*)histo_fail.GetPtr()->Clone();
  size_t found = output_file.find(std::string(".root"));
  for (unsigned int i=1; i<=Histo_pass.GetAxis(4)->GetNbins(); i++) {
    std::string newoutputname(output_file.substr(0,found));
    newoutputname+=std::string("_")+std::to_string(i)+std::string(".root");
    TFile f_out(newoutputname.c_str(),"RECREATE");
    Histo_pass.GetAxis(4)->SetRange(i,i);
    TH3D* histo_pass=(TH3D*)Histo_pass.Projection(0,1,2);
    histo_pass->SetName(Histo_pass.GetName());
    histo_pass->Write();
    delete histo_pass;
    Histo_fail.GetAxis(4)->SetRange(i,i);
    TH3D* histo_fail=(TH3D*)Histo_fail.Projection(0,1,2);
    histo_fail->SetName(Histo_fail.GetName());
    histo_fail->Write();
    delete histo_fail;
    f_out.Close();
  }
}

void saveHistogramsGen(ROOT::RDF::RResultPtr<THnT<double> > histo_pass, ROOT::RDF::RResultPtr<THnT<double> > histo_norm, std::string output_file) {
  THnD Histo_pass=*(THnD*)histo_pass.GetPtr()->Clone();
  THnD Histo_norm=*(THnD*)histo_norm.GetPtr()->Clone();
  size_t found = output_file.find(std::string(".root"));
  for (unsigned int i=1; i<=Histo_pass.GetAxis(3)->GetNbins(); i++) {
	std::string newoutputname(output_file.substr(0,found));
    newoutputname+=std::string("_")+std::to_string(i)+std::string(".root");
    TFile f_out(newoutputname.c_str(),"RECREATE");
    Histo_pass.GetAxis(3)->SetRange(i,i);
    TH2D* histo_pass=(TH2D*)Histo_pass.Projection(0,1);
    histo_pass->SetName(Histo_pass.GetName());
    histo_pass->Write();
    delete histo_pass;
    Histo_norm.GetAxis(3)->SetRange(i,i);
    TH2D* histo_norm=(TH2D*)Histo_norm.Projection(0,1);
    histo_norm->SetName(Histo_norm.GetName());
    histo_norm->Write();
    delete histo_norm;
    f_out.Close();
  }
}
