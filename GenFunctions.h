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

bool genleptoncompare (std::pair<int, float> i, std::pair<int, float> j) { 
	return (i.second>j.second);
}

int PostFSRIdx(const RVec<int> &pdgId, const RVec<int> &status, const RVec<int> &genPartIdxMother, const RVec<int> &statusFlag, const RVec<float> &pt, int Status) {
	std::vector<int> postfsrmum;
	std::vector<int> postfsrother;

	int GenPart_postFSRLepIdx1_, GenPart_postFSRLepIdx2_;

	for (unsigned int i = 0; i < pdgId.size(); i++) {
		if (genPartIdxMother[i] < 0)
			continue;
		int mompdgId = pdgId[genPartIdxMother[i]];
		if (abs(pdgId[i]) < 11 || abs(pdgId[i]) > 16)
			continue;
		if (abs(mompdgId) == 23 || abs(mompdgId) == 24) {                       //
			if ((status[i]==1)||(status[i]==2))
				if (statusFlag[i] & 1)
					postfsrmum.push_back(i);
		}
		else {
			if ((status[i]==1)||(status[i]==2))
				if (statusFlag[i] & 1)
					postfsrother.push_back(i);
		}
	}

	std::vector<std::pair<int, float> > postfsr(0);
	for (unsigned int h=0; h!=postfsrmum.size(); h++) {
		postfsr.push_back({postfsrmum[h],pt[postfsrmum[h]]});
	}
	for (unsigned int h=0; h!=postfsrother.size(); h++) {
		postfsr.push_back({postfsrother[h],pt[postfsrother[h]]});
	}
	sort(postfsr.begin(),postfsr.begin()+postfsrmum.size(),genleptoncompare);
	sort(postfsr.begin()+postfsrmum.size(),postfsr.end(),genleptoncompare);

	if (postfsr.size()==0) {
		GenPart_postFSRLepIdx1_ = -1;
		GenPart_postFSRLepIdx2_ = -1;
	}
	else if (postfsr.size()==1) {
		GenPart_postFSRLepIdx1_ = (postfsr[0]).first;
		GenPart_postFSRLepIdx2_ = -1;
	}
	else {
		GenPart_postFSRLepIdx1_ = (postfsr[0]).second > (postfsr[1]).second ? (postfsr[0]).first : (postfsr[1]).first;
		GenPart_postFSRLepIdx2_ = (postfsr[0]).second > (postfsr[1]).second ? (postfsr[1]).first : (postfsr[0]).first;
	}
	if (Status==0) return GenPart_postFSRLepIdx1_;
	else if (Status==1) return GenPart_postFSRLepIdx2_;
	return -1;
}

bool mapcompare(std::pair<double,std::pair<int,int> > i, std::pair<double,std::pair<int,int> > j) {
	return i.first<j.first;
}

void insert( std::vector<std::pair<double,std::pair<int,int> > > &cont, std::pair<double,std::pair<int,int> > value ) {
    std::vector<std::pair<double,std::pair<int,int> > >::iterator it = std::lower_bound( cont.begin(), cont.end(), value, mapcompare );
    cont.insert( it, value );
}

bool Mapcompare(std::pair<double,int> i, std::pair<double,int> j) {
	return i.first<j.first;
}

void insert( std::vector<std::pair<double,int> > &cont, std::pair<double,int> value ) {
    std::vector<std::pair<double,int> >::iterator it = std::lower_bound( cont.begin(), cont.end(), value, Mapcompare );
    cont.insert( it, value );
}

RVec<float> goodgenvalue(RVec<float> &GenPart_pt, int GenPart_postFSRLepIdx1, int GenPart_postFSRLepIdx2, RVec<float> &GenPart_eta, RVec<float> &GenPart_phi, RVec<int> &GenPart_status, RVec<int> &GenPart_pdgId) {
	RVec<float> v;
	for (auto i=0U;i < GenPart_pt.size(); ++i) {
		if (((i==GenPart_postFSRLepIdx1)||(i==GenPart_postFSRLepIdx2))&&(abs(GenPart_pdgId[i])==13)) {
			bool condition=true;
			for (auto j=0U; j < GenPart_pt.size(); ++j) {
				if (i==j) continue;
				if (GenPart_status[j]!=1) continue;
				if (!(((j==GenPart_postFSRLepIdx1)||(j==GenPart_postFSRLepIdx2))&&(abs(GenPart_pdgId[j])==13))) continue;
				TLorentzVector cand1, cand2;
				cand1.SetPtEtaPhiM(3.,GenPart_eta[i],GenPart_phi[i],0.);
				cand2.SetPtEtaPhiM(3.,GenPart_eta[j],GenPart_phi[j],0.);
				if (cand1.DeltaR(cand2)<0.3) condition=false;
			}
			if (condition) v.emplace_back(GenPart_pt[i]);
		}
	}
	return v;
}

RVec<int> goodgencharge(int GenPart_postFSRLepIdx1, int GenPart_postFSRLepIdx2, RVec<float> &GenPart_eta, RVec<float> &GenPart_phi, RVec<int> &GenPart_status, RVec<int> &GenPart_pdgId) {
	RVec<int> v;
	for (auto i=0U;i < GenPart_eta.size(); ++i) {
		if (((i==GenPart_postFSRLepIdx1)||(i==GenPart_postFSRLepIdx2))&&(abs(GenPart_pdgId[i])==13)) {
			bool condition=true;
			for (auto j=0U; j < GenPart_eta.size(); ++j) {
				if (i==j) continue;
				if (GenPart_status[j]!=1) continue;
				if (!(((j==GenPart_postFSRLepIdx1)||(j==GenPart_postFSRLepIdx2))&&(abs(GenPart_pdgId[j])==13))) continue;
				TLorentzVector cand1, cand2;
				cand1.SetPtEtaPhiM(3.,GenPart_eta[i],GenPart_phi[i],0.);
				cand2.SetPtEtaPhiM(3.,GenPart_eta[j],GenPart_phi[j],0.);
				if (cand1.DeltaR(cand2)<0.3) condition=false;
			}
			if (condition) v.emplace_back(-GenPart_pdgId[i]/abs(GenPart_pdgId[i]));
		}
	}
	return v;
}

RVec<int> goodgenidx(RVec<float> &GenPart_pt, int GenPart_postFSRLepIdx1, int GenPart_postFSRLepIdx2, RVec<float> &GenPart_eta, RVec<float> &GenPart_phi, RVec<int> &GenPart_status, RVec<int> &GenPart_pdgId) {
	RVec<int> v;
	for (auto i=0U;i < GenPart_pt.size(); ++i) {
		if (((i==GenPart_postFSRLepIdx1)||(i==GenPart_postFSRLepIdx2))&&(abs(GenPart_pdgId[i])==13)) {
			bool condition=true;
			for (auto j=0U; j < GenPart_pt.size(); ++j) {
				if (i==j) continue;
				if (GenPart_status[j]!=1) continue;
				if (!(((j==GenPart_postFSRLepIdx1)||(j==GenPart_postFSRLepIdx2))&&(abs(GenPart_pdgId[j])==13))) continue;
				TLorentzVector cand1, cand2;
				cand1.SetPtEtaPhiM(3.,GenPart_eta[i],GenPart_phi[i],0.);
				cand2.SetPtEtaPhiM(3.,GenPart_eta[j],GenPart_phi[j],0.);
				if (cand1.DeltaR(cand2)<1.) condition=false;
			}
			if (condition) v.emplace_back(i);
		}
	}
	return v;
}

RVec<Bool_t> hasGenMatchBijective(RVec<Int_t> &GenPart_pdgId, RVec<Int_t> &GenPart_status,RVec<Int_t> &GenPart_statusFlags,RVec<Float_t> &GenPart_eta,RVec<Float_t> &GenPart_phi,RVec<Float_t> &Cand_eta, RVec<Float_t> &Cand_phi, RVec<Int_t> &goodgenidx){
  RVec<Bool_t> isGenMatched;  
  RVec<int> v;
  std::vector<std::pair<double,std::pair<int,int> > > Map;
  for (unsigned int i=0; i!=Cand_eta.size(); i++) {
    v.emplace_back(-1);
  }
  for(int iCand=0;iCand<Cand_eta.size();iCand++){
    bool matchMC = false;   
    for(unsigned int iGen=0; iGen<GenPart_pdgId.size(); iGen++){
      TLorentzVector cand1, cand2;
      cand1.SetPtEtaPhiM(3.,GenPart_eta[iGen],GenPart_phi[iGen],0.);
      cand2.SetPtEtaPhiM(3.,Cand_eta[iCand],Cand_phi[iCand],0.);
      float deltar=cand1.DeltaR(cand2);
      if ( abs(GenPart_pdgId[iGen])==13 &&  GenPart_status[iGen] == 1 && (GenPart_statusFlags[iGen] & 1)  ) {
	if (deltar < 0.3){
      insert(Map,{deltar,{iGen,iCand}});
	  //truePt     = GenPart_pt[ii];
	  //trueEta    = GenPart_eta[ii];
	  //trueCharge = GenPart_charge[ii];
	} 
      }
    }
	std::vector<int> dropind, dropgen;
    for (unsigned int i=0; i!=Map.size(); i++) {
      auto it = Map.begin();
      std::advance(it,i);
      int genidx=it->second.first, idx=it->second.second;
      if (std::find(dropind.begin(), dropind.end(), idx) != dropind.end()) continue;
      if (std::find(dropgen.begin(), dropgen.end(), genidx) != dropgen.end()) continue;
      v[idx]=goodgenidx[genidx];
      dropind.emplace_back(idx);
      dropgen.emplace_back(genidx);
	}
  }
  for (unsigned int i=0; i!=v.size(); i++) isGenMatched.emplace_back(v[i]>-1);
  return isGenMatched;
}

RVec<bool> goodmuonreco(RVec<float> &goodgeneta, RVec<float> &goodgenphi, RVec<float> &MergedStandAloneMuon_pt, RVec<float> &MergedStandAloneMuon_eta, RVec<float> &MergedStandAloneMuon_phi) {
  RVec<bool> v;
  for (auto i=0U;i < goodgeneta.size(); ++i) {
    std::vector<std::pair<double,int> > Map;
    for (auto j=0U; j < MergedStandAloneMuon_eta.size(); ++j) {
        TLorentzVector cand, cand2;
      cand.SetPtEtaPhiM(5.,goodgeneta[i],goodgenphi[i],0.);
      cand2.SetPtEtaPhiM(5.,MergedStandAloneMuon_eta[j],MergedStandAloneMuon_phi[j],0.);
      insert(Map,{cand.DeltaR(cand2),j});
    }
    if ((Map.size()>0)&&(Map.begin()->first<0.3)) {
      if ((MergedStandAloneMuon_pt[Map.begin()->second]>15)) v.emplace_back(1);
      else v.emplace_back(0);
    }
    else v.emplace_back(0); 
  }
  return v;
}

RVec<bool> goodmuonglobal(RVec<float> &goodgeneta, RVec<float> &goodgenphi, RVec<float> &Muon_pt, RVec<float> &Muon_eta, RVec<float> &Muon_phi, RVec<bool> &Muon_isGlobal, RVec<float> &Muon_standalonePt, RVec<float> &Muon_standaloneEta, RVec<float> &Muon_standalonePhi) {
  RVec<bool> v;
  for (auto i=0U;i < goodgeneta.size(); ++i) {
    std::vector<std::pair<double,int> > Map;
    for (auto j=0U; j < Muon_eta.size(); ++j) {
      if (!Muon_isGlobal[i]) continue;
      TLorentzVector cand, cand2;
      cand.SetPtEtaPhiM(5.,goodgeneta[i],goodgenphi[i],0.);
      cand2.SetPtEtaPhiM(5.,Muon_eta[j],Muon_phi[j],0.);
      insert(Map,{cand.DeltaR(cand2),j});
    }
    if ((Map.size()>0)&&(Map.begin()->first<0.3)) {
      TLorentzVector cand2,cand3;
      cand2.SetPtEtaPhiM(5.,Muon_eta[Map.begin()->second],Muon_phi[Map.begin()->second],0.);
      cand3.SetPtEtaPhiM(5.,Muon_standaloneEta[Map.begin()->second],Muon_standalonePhi[Map.begin()->second],0.);
      if ((Muon_pt[Map.begin()->second]>15)&&(Muon_standalonePt[Map.begin()->second]>15.)&&(cand2.DeltaR(cand3)<0.3)) v.emplace_back(1);
      else v.emplace_back(0);
    }
    else v.emplace_back(0); 
  }
  return v;
}

RVec<bool> goodmuonidip(RVec<float> &goodgeneta, RVec<float> &goodgenphi, RVec<float> &Muon_pt, RVec<float> &Muon_eta, RVec<float> &Muon_phi, RVec<bool> &Muon_isGlobal, RVec<float> &Muon_standalonePt, RVec<float> &Muon_standaloneEta, RVec<float> &Muon_standalonePhi, RVec<float> &Muon_dxybs, RVec<bool> &Muon_isMedium) {
  RVec<bool> v;
  for (auto i=0U;i < goodgeneta.size(); ++i) {
    std::vector<std::pair<double,int> > Map;
    for (auto j=0U; j < Muon_eta.size(); ++j) {
      if (!Muon_isGlobal[i]) continue;
      TLorentzVector cand, cand2;
      cand.SetPtEtaPhiM(5.,goodgeneta[i],goodgenphi[i],0.);
      cand2.SetPtEtaPhiM(5.,Muon_eta[j],Muon_phi[j],0.);
      insert(Map,{cand.DeltaR(cand2),j});
    }
    if ((Map.size()>0)&&(Map.begin()->first<0.3)) {
      TLorentzVector cand2,cand3;
      cand2.SetPtEtaPhiM(5.,Muon_eta[Map.begin()->second],Muon_phi[Map.begin()->second],0.);
      cand3.SetPtEtaPhiM(5.,Muon_standaloneEta[Map.begin()->second],Muon_standalonePhi[Map.begin()->second],0.);
      if ((Muon_pt[Map.begin()->second]>15)&&(Muon_standalonePt[Map.begin()->second]>15.)&&(cand2.DeltaR(cand3)<0.3)&&(abs(Muon_dxybs[Map.begin()->second])<0.05)&&(Muon_isMedium[Map.begin()->second])) v.emplace_back(1);
      else v.emplace_back(0);
    }
    else v.emplace_back(0); 
  }
  return v;
}

RVec<bool> goodmuontrigger(RVec<float> &goodgeneta, RVec<float> &goodgenphi, RVec<float> &Muon_pt, RVec<float> &Muon_eta, RVec<float> &Muon_phi, RVec<bool> &Muon_isGlobal, RVec<float> &Muon_standalonePt, RVec<float> &Muon_standaloneEta, RVec<float> &Muon_standalonePhi, RVec<float> &Muon_dxybs, RVec<bool> &Muon_isMedium, RVec<bool> &isTriggeredMuon) {
  RVec<bool> v;
  for (auto i=0U;i < goodgeneta.size(); ++i) {
    std::vector<std::pair<double,int> > Map;
    for (auto j=0U; j < Muon_eta.size(); ++j) {
      if (!Muon_isGlobal[i]) continue;
      TLorentzVector cand, cand2;
      cand.SetPtEtaPhiM(5.,goodgeneta[i],goodgenphi[i],0.);
      cand2.SetPtEtaPhiM(5.,Muon_eta[j],Muon_phi[j],0.);
      insert(Map,{cand.DeltaR(cand2),j});
    }
    if ((Map.size()>0)&&(Map.begin()->first<0.3)) {
      TLorentzVector cand2,cand3;
      cand2.SetPtEtaPhiM(5.,Muon_eta[Map.begin()->second],Muon_phi[Map.begin()->second],0.);
      cand3.SetPtEtaPhiM(5.,Muon_standaloneEta[Map.begin()->second],Muon_standalonePhi[Map.begin()->second],0.);
      if ((Muon_pt[Map.begin()->second]>15)&&(Muon_standalonePt[Map.begin()->second]>15.)&&(cand2.DeltaR(cand3)<0.3)&&(abs(Muon_dxybs[Map.begin()->second])<0.05)&&(Muon_isMedium[Map.begin()->second])&&(isTriggeredMuon[Map.begin()->second])) v.emplace_back(1);
      else v.emplace_back(0);
    }
    else v.emplace_back(0); 
  }
  return v;
}

RVec<bool> goodmuonisolation(RVec<float> &goodgeneta, RVec<float> &goodgenphi, RVec<float> &Muon_pt, RVec<float> &Muon_eta, RVec<float> &Muon_phi, RVec<bool> &Muon_isGlobal, RVec<float> &Muon_standalonePt, RVec<float> &Muon_standaloneEta, RVec<float> &Muon_standalonePhi, RVec<float> &Muon_dxybs, RVec<bool> &Muon_isMedium, RVec<bool> &isTriggeredMuon, RVec<float> &Muon_pfRelIso04_all) {
  RVec<bool> v;
  for (auto i=0U;i < goodgeneta.size(); ++i) {
    std::vector<std::pair<double,int> > Map;
    for (auto j=0U; j < Muon_eta.size(); ++j) {
      if (!Muon_isGlobal[i]) continue;
      TLorentzVector cand, cand2;
      cand.SetPtEtaPhiM(5.,goodgeneta[i],goodgenphi[i],0.);
      cand2.SetPtEtaPhiM(5.,Muon_eta[j],Muon_phi[j],0.);
      insert(Map,{cand.DeltaR(cand2),j});
    }
    if ((Map.size()>0)&&(Map.begin()->first<0.3)) {
      TLorentzVector cand2,cand3;
      cand2.SetPtEtaPhiM(5.,Muon_eta[Map.begin()->second],Muon_phi[Map.begin()->second],0.);
      cand3.SetPtEtaPhiM(5.,Muon_standaloneEta[Map.begin()->second],Muon_standalonePhi[Map.begin()->second],0.);
      if ((Muon_pt[Map.begin()->second]>15)&&(Muon_standalonePt[Map.begin()->second]>15.)&&(cand2.DeltaR(cand3)<0.3)&&(abs(Muon_dxybs[Map.begin()->second])<0.05)&&(Muon_isMedium[Map.begin()->second])&&(isTriggeredMuon[Map.begin()->second])&&(Muon_pfRelIso04_all[Map.begin()->second])) v.emplace_back(1);
      else v.emplace_back(0);
    }
    else v.emplace_back(0); 
  }
  return v;
}

RVec<float> postFSRgenzqtprojection(RVec<float> &goodgenpt, RVec<float> &goodgeneta, RVec<float> &goodgenphi) {
  RVec<float> v;
  for (auto i=0U; i<goodgenpt.size(); i++) {
    TLorentzVector probe;
    probe.SetPtEtaPhiM(goodgenpt[i],goodgeneta[i],goodgenphi[i],0.);
    for (auto j=0U; j<goodgenpt.size(); j++) {
      if (i==j) continue;
      TLorentzVector tag;
      tag.SetPtEtaPhiM(goodgenpt[j],goodgeneta[j],goodgenphi[j],0.);
      TVector3 Tag(tag.Px(),tag.Py(),0.), Probe(probe.Px(),probe.Py(),0.);
      v.emplace_back((Tag+Probe).Dot(Probe)/sqrt(Probe.Dot(Probe)));
    }
  }
  return v;
}
