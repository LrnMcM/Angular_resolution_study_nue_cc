import uproot
import awkward as ak

fname = "NuE_CC_2603_noFar_noShield_Pandora_Cheated_PaigeFixProper.root"

file = uproot.open(fname)
t = file["LArRecoND"]

print("TREE:")
print(t)

print("\nBRANCH TYPES:")
print("mcPDG:", t["mcPDG"].interpretation)
print("mcPx: ", t["mcPx"].interpretation)
print("mcPy: ", t["mcPy"].interpretation)
print("mcPz: ", t["mcPz"].interpretation)

pdg = t["mcPDG"].array(library="ak")
px = t["mcPx"].array(library="ak")
py = t["mcPy"].array(library="ak")
pz = t["mcPz"].array(library="ak")

print("\nAWKWARD TYPES:")
print("mcPDG:", ak.type(pdg))
print("mcPx: ", ak.type(px))
print("mcPy: ", ak.type(py))
print("mcPz: ", ak.type(pz))

print("\nNUMBER OF EVENTS:")
print(len(pdg))

print("\nFIRST 5 EVENTS:")

for i in range(min(5, len(pdg))):

    print(f"\nEVENT {i}")

    print("PDG =", pdg[i])
    print("Px  =", px[i])
    print("Py  =", py[i])
    print("Pz  =", pz[i])
