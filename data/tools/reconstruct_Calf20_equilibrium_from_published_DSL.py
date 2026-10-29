from pathlib import Path
import csv, math
R=8.314462618
temps=[303.15,313.15,323.15,333.15,353.15,373.15]
pressures=[0.003,0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.3,0.5,0.75,1,1.5,2,3,4,5]
P={("CO2","b"):(2.387,5.52e-7,35.06e3),("CO2","d"):(3.271,5.19e-8,28.95e3),
   ("N2","b"):(2.387,8.14e-7,17.96e3),("N2","d"):(3.271,0.0,0.0)}
rows=[]
for gas in ["CO2","N2"]:
  for T in temps:
    for pbar in pressures:
      C=pbar*1e5/(R*T); q=0.0
      for site in ["b","d"]:
        qs,b0,Q=P[(gas,site)]
        b=b0*math.exp(Q/(R*T)) if b0 else 0
        q += qs*b*C/(1+b*C) if b else 0
      rows.append([gas,T,pbar,q])
out=Path(__file__).with_name("regenerated_numeric_core.csv")
with out.open("w",newline="",encoding="utf-8") as f:
  w=csv.writer(f); w.writerow(["adsorbate","temperature_K","pressure_bar","loading_mol_kg"]); w.writerows(rows)
print(out)
