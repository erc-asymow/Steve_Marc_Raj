#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "TCanvas.h"
#include "TH1D.h"
#include "TLatex.h"
#include "Math/Vector4D.h"
#include "TStyle.h"
#include <string>

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


float deltaPhi(float phi1, float phi2) {                                                        
  float result = phi1 - phi2;
  while (result > float(M_PI)) result -= float(2*M_PI);
  while (result <= -float(M_PI)) result += float(2*M_PI);
  return result;
}

float deltaR2(float eta1, float phi1, float eta2, float phi2) {
  float deta = std::abs(eta1-eta2);
  float dphi = deltaPhi(phi1,phi2);
  return deta*deta + dphi*dphi;
}

float deltaR(float eta1, float phi1, float eta2, float phi2) {
  return std::sqrt(deltaR2(eta1,phi1,eta2,phi2));
}

RVec<Int_t> CreateProbes_Muon(RVec<Float_t> &Muon_pt, RVec<Float_t> &Muon_standalonePt,RVec<Float_t> &Muon_eta,RVec<Float_t> &Muon_phi, RVec<Float_t> &Muon_standaloneEta, RVec<Float_t> &Muon_standalonePhi, RVec<Int_t> &Muon_charge, RVec<Bool_t> &Muon_mediumId, RVec<Float_t> &Muon_dxybs,RVec<Bool_t> &Muon_isGlobal){
  RVec<Int_t> Probe_Muons;
  for(int i=0;i<Muon_pt.size();i++){
    if(Muon_pt.at(i) < 15 || Muon_standalonePt.at(i) < 15 || abs(Muon_eta.at(i)) > 2.4 || !Muon_isGlobal.at(i) || deltaR(Muon_eta.at(i),Muon_phi.at(i),Muon_standaloneEta.at(i),Muon_standalonePhi.at(i))>0.3) continue;
    //Probe probe_muon;
    //probe_muon.pt = Muon_pt.at(i);
    //probe_muon.eta = Muon_eta.at(i);
    //probe_muon.phi = Muon_phi.at(i);
    //probe_muon.charge = Muon_charge.at(i);
    //probe_muon.original_index = i;
    Probe_Muons.push_back(i);
  }
  return Probe_Muons;
}

RVec<Int_t> CreateProbes_Track(RVec<Float_t> &Track_pt, RVec<Float_t> &Track_eta,RVec<Float_t> &Track_phi,RVec<Int_t> &Track_charge, RVec<Int_t> &Track_trackOriginalAlgo){

  RVec<Int_t> Probe_Tracks;
  for(int i=0;i<Track_pt.size();i++){
    if(Track_pt.at(i) < 15. || Track_trackOriginalAlgo.at(i) == 13 || Track_trackOriginalAlgo.at(i) == 14) continue;
    //Probe probe_track;
    //probe_track.pt = Track_pt.at(i);
    //probe_track.eta = Track_eta.at(i);
    //probe_track.phi = Track_phi.at(i);
    //probe_track.charge = Track_charge.at(i);
    //probe_track.original_index = i;
    Probe_Tracks.push_back(i);
  }
  return Probe_Tracks;

}

RVec<Int_t> CreateProbes_MergedStandMuons(RVec<Float_t> &MergedStandAloneMuon_pt, RVec<Float_t> &MergedStandAloneMuon_eta, RVec<Float_t> &MergeStandAloneMuon_phi){

  RVec<Int_t> Probe_Stand;
  for(int i=0;i<MergedStandAloneMuon_pt.size();i++){
    if(MergedStandAloneMuon_pt.at(i) < 15.) continue;
    //Probe probe_track;
    //probe_track.pt = Track_pt.at(i);
    //probe_track.eta = Track_eta.at(i);
    //probe_track.phi = Track_phi.at(i);
    //probe_track.charge = Track_charge.at(i);
    //probe_track.original_index = i;
    Probe_Stand.push_back(i);
  }
  return Probe_Stand;

}

RVec<std::pair<int,int>> CreateTPPair(RVec<Int_t> &Muon_charge, RVec<Int_t> &isTag, RVec<Bool_t> &isTriggeredMuon, RVec<Bool_t> &isGenMatchedMuon, RVec<Int_t> &Probe_Candidates, RVec<Bool_t> &isGenMatchedProbe ){
  RVec<std::pair<int,int>> TP_pairs;
  for(int iLep1=0; iLep1<Muon_charge.size();iLep1++){
    //if(!isInAcceptance[iLep1]) continue;
    if(!isTag[iLep1]) continue;
    if(!isTriggeredMuon[iLep1]) continue;
    if(!isGenMatchedMuon[iLep1]) continue;




    for(int iLep2=0; iLep2<Probe_Candidates.size(); iLep2++){
      int probe_candidate = Probe_Candidates.at(iLep2);
      //if (iLep2==iLep1) continue;
      //if(!isProbe[iLep2]) continue;
      //if(!isTriggeredMuon[iLep2]) continue;
      if(!isGenMatchedProbe[probe_candidate]) continue;
      //#if(Muon_charge[iLep1] == Muon_charge[iLep2]) continue;

      std::pair<int,int> TP_pair = std::make_pair(iLep1,probe_candidate); 
      TP_pairs.push_back(TP_pair);
    }

  }
  return TP_pairs;
}

RVec<Bool_t> hasStandAloneOrGlobalMatch(RVec<Float_t> &Track_eta, RVec<Float_t> &Track_phi,RVec<Float_t> &Muon_eta, RVec<Float_t> &Muon_phi,RVec<Bool_t> &Muon_isStandalone,RVec<Bool_t> &Muon_isGlobal){
  RVec<Int_t> hasStandAloneOrGlobalMatch;
  for(int iTrack=0;iTrack<Track_eta.size();iTrack++){
    bool has_match = false;
    for (int iMuon=0; iMuon<Muon_eta.size(); ++iMuon){
      if (!(Muon_isStandalone[iMuon] || Muon_isGlobal[iMuon])) continue;
      if (deltaR(Muon_eta[iMuon], Muon_phi[iMuon], Track_eta[iTrack], Track_phi[iTrack]) < 0.01) {has_match = 1; break;}
    } 
    hasStandAloneOrGlobalMatch.push_back(has_match);
  }
  return hasStandAloneOrGlobalMatch;
}


RVec<Float_t> trackMuonDR(RVec<Float_t> &Track_eta, RVec<Float_t> &Track_phi,RVec<Float_t> &Muon_eta,RVec<Float_t> &Muon_phi){
  RVec<Float_t> trackMuonDR; 
  for(int iTrack=0;iTrack<Track_eta.size();iTrack++){
    float dr = 999.;
    float tmp_dr  = 999.;

    for (unsigned int iMuon=0; iMuon<Muon_eta.size(); ++iMuon){
      tmp_dr  = deltaR(Muon_eta.at(iMuon), Muon_phi.at(iMuon), Track_eta.at(iTrack), Track_phi.at(iTrack));
      if (tmp_dr < dr) dr = tmp_dr;
    }
    trackMuonDR.push_back(dr);
  }  
  return trackMuonDR;
}


RVec<Float_t> trackStandaloneDR(RVec<Float_t> &Track_eta, RVec<Float_t> &Track_phi,RVec<Float_t> &Muon_standaloneEta,RVec<Float_t> &Muon_standalonePhi){
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

RVec<Bool_t> hasTriggerMatch(RVec<Float_t> &Muon_eta, RVec<Float_t> &Muon_phi, RVec<Int_t> &TrigObj_id,RVec<Float_t> &TrigObj_pt,RVec<Float_t> & TrigObj_l1pt, RVec<Float_t> & TrigObj_l2pt, RVec<Int_t> &TrigObj_filterBits, RVec<Float_t> &TrigObj_eta, RVec<Float_t> &TrigObj_phi){
  RVec<Bool_t> TriggerMatch;
  for (int iMuon = 0; iMuon<Muon_eta.size(); iMuon++ ){
    bool hasTrigMatch = false;
    for (unsigned int iTrig=0; iTrig<TrigObj_id.size(); ++iTrig){
      if (TrigObj_id[iTrig]  != 13 ) continue;
      if (TrigObj_pt[iTrig]   < 24.) continue;
      if (TrigObj_l1pt[iTrig] < 22.) continue;
      if (! (( TrigObj_filterBits[iTrig] & 8) || (TrigObj_l2pt[iTrig] > 10. && (TrigObj_filterBits[iTrig] & 2) )) ) continue;
      if (deltaR(Muon_eta[iMuon], Muon_phi[iMuon], TrigObj_eta[iTrig], TrigObj_phi[iTrig]) < 0.3) {
	hasTrigMatch = true;
	break;
      }
    }
    TriggerMatch.push_back(hasTrigMatch);
  }
  return TriggerMatch;
}

RVec<Bool_t> hasGenMatch(RVec<Int_t> &GenPart_pdgId, RVec<Int_t> &GenPart_status,RVec<Int_t> &GenPart_statusFlags,RVec<Float_t> &GenPart_eta,RVec<Float_t> &GenPart_phi,RVec<Float_t> &Cand_eta, RVec<Float_t> &Cand_phi){
  RVec<Bool_t> isGenMatched;  
  for(int iCand=0;iCand<Cand_eta.size();iCand++){
    bool matchMC = false;   
    float mcmatch_tmp_dr = 999.;
    for(unsigned int iGen=0; iGen<GenPart_pdgId.size(); iGen++){
      if ( abs(GenPart_pdgId[iGen])==13 &&  GenPart_status[iGen] == 1 && (GenPart_statusFlags[iGen] & 1)  ) {
	if (deltaR(Cand_eta[iCand], Cand_phi[iCand], GenPart_eta[iGen], GenPart_phi[iGen]) < mcmatch_tmp_dr){
	  mcmatch_tmp_dr  = deltaR(Cand_eta[iCand], Cand_phi[iCand], GenPart_eta[iGen], GenPart_phi[iGen]);
	  //truePt     = GenPart_pt[ii];
	  //trueEta    = GenPart_eta[ii];
	  //trueCharge = GenPart_charge[ii];
	} 
      }
    }
    if (mcmatch_tmp_dr < 0.1) matchMC = true;
    isGenMatched.push_back(matchMC);
  }
  return isGenMatched;
}



RVec<Float_t> getTPmass(RVec<std::pair<int,int>> TPPairs, RVec<Float_t> &Muon_pt,RVec<Float_t> &Muon_eta,RVec<Float_t> &Muon_phi, RVec<Float_t> &Cand_pt, RVec<Float_t> &Cand_eta, RVec<Float_t> &Cand_phi){
  RVec<Float_t> TPMass;
  for (int i=0;i<TPPairs.size();i++){
    std::pair<int,int> TPPair = TPPairs.at(i);
    int tag_index = TPPair.first;
    int probe_index = TPPair.second;
    TLorentzVector tagLep(0,0,0,0);
    tagLep.SetPtEtaPhiM(Muon_pt[tag_index],Muon_eta[tag_index],Muon_phi[tag_index],0);
    TLorentzVector probeLep(0,0,0,0);
    probeLep.SetPtEtaPhiM(Cand_pt[probe_index],Cand_eta[probe_index],Cand_phi[probe_index],0);
    TPMass.push_back((tagLep+probeLep).M());        
  }
  return TPMass;
}


RVec<Float_t> getVariables(RVec<std::pair<int,int>> TPPairs, RVec<Float_t> &Cand_variable, float option /*1 for tag and 2 for probe*/){
  RVec<Float_t> Variables;
  for (int i=0;i<TPPairs.size();i++){
    std::pair<int,int> TPPair = TPPairs.at(i);
    float variable; 
    if (option==1) variable = Cand_variable.at(TPPair.first);
    else if (option==2) variable = Cand_variable.at(TPPair.second);
    Variables.push_back(variable);
  }
  return Variables;
}

RVec<Int_t> getVariables(RVec<std::pair<int,int>> TPPairs, RVec<Int_t> &Cand_variable,float option /*1 for tag and 2 for probe*/){
  RVec<Int_t> Variables;
  for (int i=0;i<TPPairs.size();i++){
    std::pair<int,int> TPPair = TPPairs.at(i);
    int variable;
    if(option==1) variable = Cand_variable.at(TPPair.first);
    else if (option==2) variable = Cand_variable.at(TPPair.second);
    Variables.push_back(variable);
  }
  return Variables;
}

RVec<Bool_t> getVariables(RVec<std::pair<int,int>> TPPairs, RVec<Bool_t> &Cand_variable,float option /*1 for tag and 2 for probe*/){
  RVec<Bool_t> Variables;
  for (int i=0;i<TPPairs.size();i++){
    std::pair<int,int> TPPair = TPPairs.at(i);
    bool variable;
    if(option==1) variable = Cand_variable.at(TPPair.first);
    else if (option==2) variable = Cand_variable.at(TPPair.second);
    Variables.push_back(variable);
  }
  return Variables;
}

RVec<Float_t> zqtprojection(RVec<std::pair<int,int>> &TPPairs, RVec<Float_t> &Muon_pt, RVec<Float_t> &Muon_eta, RVec<Float_t> &Muon_phi) {
  RVec<Float_t> v;
  for (int i=0;i<TPPairs.size();i++){
    std::pair<int,int> TPPair = TPPairs.at(i);
    TLorentzVector tag, probe;
    tag.SetPtEtaPhiM(Muon_pt[TPPair.first],Muon_eta[TPPair.first],Muon_phi[TPPair.first],0.);
    probe.SetPtEtaPhiM(Muon_pt[TPPair.second],Muon_eta[TPPair.second],Muon_phi[TPPair.second],0.);
    TVector3 Tag(tag.Px(),tag.Py(),0.), Probe(probe.Px(), probe.Py(), 0.);
    v.emplace_back((Tag+Probe).Dot(Probe)/sqrt(Probe.Dot(Probe)));
  }
  return v;
}

RVec<Bool_t> isOS(RVec<std::pair<int,int>> TPPairs,RVec<Int_t> Muon_charge, RVec<Int_t> Cand_charge){
  RVec<Bool_t> isOS;
  for (int i=0;i<TPPairs.size();i++){
    std::pair<int,int> TPPair = TPPairs.at(i);
    int  tag_index = TPPair.first;
    int probe_index = TPPair.second;
    bool os = false;
    if (Muon_charge[tag_index] != Cand_charge[probe_index]) os = true; //initially ==. We want this to return true for opposite sign pairs, right?
    isOS.push_back(os);
  }
  return isOS;
}

RVec<Bool_t> Probe_isGlobal(RVec<std::pair<int,int>> &TPPairs, RVec<Int_t> &MergedStandAloneMuon_extraIdx, RVec<Int_t> &Muon_standaloneExtraIdx, RVec<Bool_t> &Muon_isGlobal, RVec<Float_t> &Muon_pt, RVec<Float_t> &Muon_eta, RVec<Float_t> &Muon_phi, RVec<Float_t> &Muon_standalonePt, RVec<Float_t> &Muon_standaloneEta, RVec<Float_t> &Muon_standalonePhi) {
  RVec<Bool_t> isGlobal;
  for (auto i=0U; i<TPPairs.size(); i++) {
    std::pair<int,int> TPPair = TPPairs.at(i);
    int  tag_index = TPPair.first;
    int probe_index = TPPair.second;
    bool condition=false;
    for (auto j=0U; j<Muon_standaloneExtraIdx.size(); j++) {
      if ((MergedStandAloneMuon_extraIdx[probe_index]==Muon_standaloneExtraIdx[j])&&(Muon_isGlobal[j])&&(Muon_pt[j]>15.)&&(Muon_standalonePt[j]>15.)&&(deltaR(Muon_eta[j],Muon_phi[j],Muon_standaloneEta[j],Muon_standalonePhi[j]))) condition=true;
    }
    isGlobal.push_back(condition);
  }
  return isGlobal;
}


float clipGenWeight(float genWeight){
  float sign =  genWeight/ abs(genWeight);
  return sign;
}

RVec<Bool_t> createTrues(int size){
   RVec<Bool_t> Trues(size,true);
   return Trues;
}


float puw_2016(int nTrueInt, int period=0)
{ 
  // inclusive
  float _puw2016_nTrueInt_36fb[100] = {0.6580788810596797, 0.4048055020122533, 0.8615727417125882, 0.78011444438815, 0.7646792294649241, 0.4428435234418468, 0.21362966755761287, 0.1860690038399781, 0.26922404664693694, 0.3391088547725875, 0.4659907891982563, 0.6239096711946419, 0.7379729344299205, 0.8023612182327676, 0.8463429262728119, 0.8977203788568622, 0.9426445582538644, 0.9730606480643851, 0.9876071912898365, 0.9967809848428801, 1.0101217492530405, 1.0297182331139065, 1.0495800245321394, 1.062993596017779, 1.0719371123276982, 1.0807330103942534, 1.0874070624201613, 1.092162599537252, 1.1002282592549013, 1.1087757206073987, 1.1145692898936854, 1.1177137330250873, 1.1227674788957978, 1.127920444541154, 1.1304392102417709, 1.1363304087643262, 1.1484550576573724, 1.1638015070216376, 1.1833470331674814, 1.2079950937281854, 1.2274834937176624, 1.2444596857655403, 1.269207386640866, 1.2896345429784277, 1.3441092120892817, 1.318565807546743, 1.3245909554149022, 1.2438485820227307, 1.1693858006040387, 0.9959473986900877, 0.8219068296672665, 0.6541529992036196, 0.4693498259046716, 0.32997121176496014, 0.31310467423384075, 0.23393084684715088, 0.2693460933531134, 0.2514663792161372, 0.3548113113715822, 0.5892264892080126, 0.6554126787320194, 0.7565044847439312, 1.0033865725393603, 1.4135667773217673, 0.8064239843371759, 1.096753894690768, 1.0, 1.0, 1.0, 1.0, 1.0, 1.409161148159025, 2.436803424843253, 1.0, 1.0, 1.5123015283558174, 1.0, 1.0, 1.0, 2.184264934077207, 0.44722379540985835, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  // preVFP
  float _puw2016_nTrueInt_BtoF[100] = {0.017867211276858426, 0.25019181678340197, 0.7305862075344697, 0.8203306355218927, 0.8568507658433443, 0.5021252568573517, 0.26598094658249777, 0.2525594126097894, 0.43091072564779115, 0.5816151668719733, 0.7400719694832786, 0.8635841784600747, 0.9396947027327512, 0.9867166485834531, 1.020291825978338, 1.058677896425578, 1.0951610451340548, 1.1270599475402712, 1.149199627966607, 1.161062844833281, 1.1570323009561523, 1.1408277403854514, 1.1237795117739837, 1.1080566938230716, 1.0897761375745778, 1.064304242937247, 1.0256371301024838, 0.974592047960344, 0.919208105883519, 0.8614029922190752, 0.80250010071568, 0.7447070188841107, 0.691304388080654, 0.6402835441510876, 0.5891723690900388, 0.5400972141379156, 0.49313582436893444, 0.4463956167604181, 0.400690015469447, 0.3570993263445747, 0.313769141057574, 0.272973076482914, 0.23756151770204517, 0.2052480930611912, 0.18165295449088267, 0.15154171165962477, 0.13020442654298184, 0.1060542785336088, 0.08931962151840153, 0.07320668426537356, 0.06699868236915886, 0.07342602448799117, 0.09022522247731071, 0.1221948811200131, 0.21751127388127575, 0.26185148964193156, 0.3993515054053398, 0.4259481886684673, 0.6348999885572649, 1.0768669803515931, 1.2073680188570837, 1.3977538479690192, 1.8560151439282793, 2.6159138791249634, 1.492621348422977, 2.030155370346692, 1.0, 1.0, 1.0, 1.0, 1.0, 2.608617469665849, 4.5109759375776735, 1.0, 1.0, 2.7995528359963338, 1.0, 1.0, 1.0, 4.043483083032464, 0.8278949263309181, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  // postVFP
  float _puw2016_nTrueInt_GtoH[100] = {1.5094769667755739, 0.5678507573322055, 1.0105884486266141, 0.7259960323564019, 0.6619453439299897, 0.37363396015230216, 0.15239333007202369, 0.10791755923476429, 0.07918472234473897, 0.05419619726622047, 0.14361053792634917, 0.34283985083406093, 0.5010734484221103, 0.5859542596067007, 0.6417013874657507, 0.7081679228576429, 0.7630928885638684, 0.7927011423655624, 0.7977355318365416, 0.804189334635704, 0.8379574350304874, 0.8991008934966008, 0.9625166892724227, 1.0100064501247878, 1.0501599728932576, 1.0991686204042044, 1.159807646806213, 1.230197853416741, 1.3121394462817406, 1.3989162630341463, 1.4799356011552773, 1.5557799985845535, 1.6310266488456922, 1.7003552120673373, 1.7676611780885065, 1.8405616412980672, 1.9250530644650858, 2.010419966610567, 2.10422235818776, 2.2195187847594804, 2.2997062399954857, 2.395010762430828, 2.4803953120930684, 2.5918893616537275, 2.7047108284644694, 2.71073345533101, 2.6519181014415363, 2.6277548910588657, 2.3664596415368297, 2.1655717980383105, 1.8274047657200394, 1.3313115741327561, 0.9113129353116672, 0.5869561379336159, 0.4284308024180917, 0.20595003106437168, 0.10999572232736675, 0.049536718778780756, 0.035006854789963834, 0.012922619333388894, 0.009214760376410974, 0.009219346474718641, 0.0015371306555079958, 0.001800361515424057, 0.0003153024223498953, 0.00010484990280832637, 1.0, 9.561632218479775e-05, 4.4263424947054785e-05, 1.0, 1.0, 8.750970959266416e-06, 1.9850705244600614e-06, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0235991128869821e-08, 1.3636783092893852e-09, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};


  if (nTrueInt<100) {
    if      (period==0) return _puw2016_nTrueInt_36fb[nTrueInt]; 
    else if (period==1) return _puw2016_nTrueInt_BtoF[nTrueInt]; 
    else if (period==2) return _puw2016_nTrueInt_GtoH[nTrueInt]; 
    else { std::cout<< " you are giving a wrong period to the pileup weight function, returning 0 for the events!!!!" << std::endl; return 0.;}
  }
  else  return 1.;
}


class JsonHelper {
public:
  using pair_t = std::pair<unsigned int, unsigned int>;
  using jsonmap_t = std::unordered_map<unsigned int, std::vector<pair_t>>;
  
  JsonHelper(const std::vector<unsigned int> &runs, const std::vector<unsigned int> &firstlumis, const std::vector<unsigned int> &lastlumis) :
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
      auto const pairit = std::lower_bound(pairs.begin(), pairs.end(), lumi, [](const pair_t &pair, unsigned int val) { return pair.second < val; } );
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

void saveHistograms(ROOT::RDF::RResultPtr<THnT<double> > histo_pass, ROOT::RDF::RResultPtr<THnT<double> > histo_fail, std::string output_file) {
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
