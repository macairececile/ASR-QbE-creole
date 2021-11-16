from argparse import ArgumentParser, RawTextHelpFormatter

# Create a text file with the .tex format which makes a parallel between HYP et HYP of two versions.
# output file containing:
# Ref: & donk kom \hl{e} \\
# Hyp\_xlsr: & donk kom \\
# Ref: & donk kom e \\
# Hyp\_LeBenchmark: & donk kom e \\
# Both latex should have the same data.

def create_latex(arguments):
    c_final = []
    with open(arguments.latex_no_lm, 'r') as f1:
        content = f1.readlines()
    with open(arguments.latex_lm, 'r') as f2:
        content2 = f2.readlines()
    for i in range(0, len(content), 3):
        c_final.append(content[i])
        c_final.append(content[i + 1])
        c_final.append(content2[i])
        c_final.append(content2[i + 1])
        c_final.append(content[i + 2])
    with open('latex_final_lm.txt', 'a') as final_f:
        for el in c_final:
            final_f.write(el)


if __name__ == '__main__':
    parser = ArgumentParser(description="Create a latex file which combines data from two latex files.", formatter_class=RawTextHelpFormatter)

    parser.add_argument('--first_latex', type=str, required=True,
                       help="Txt file with results not using LM.")
    parser.add_argument('--second_latex', type=str, required=True,
                       help="Txt file with results using LM.")
    parser.set_defaults(func=create_latex)

    args = parser.parse_args()
    args.func(args)
