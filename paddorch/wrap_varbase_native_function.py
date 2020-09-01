import os,sys

if __name__ == '__main__':
    for line in open(r"paddle\fluid\core_avx\VarBase.py"):
        if "def " in line:
            comps=line.split("(")
            if "self" not in line:
                line2=comps[0].replace("def ", "    return dygraph.core.VarBase.")+"("+comps[1].replace(":","")
            else:
                line2=comps[0].replace("def ", "    return self.valbase.")+"("+ (comps[1].split("self")[1].replace(":",""))
            print(line+"\n"+line2.replace("(,","("))