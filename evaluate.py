# -*- coding: utf-8 -*-
# Adapt from https://github.com/alvations/bayesline-DSL/


import io
import sys
from collections import Counter

red = '\033[01;31m'
native = '\033[m'

def err_msg(txt):
    return red + txt + native

def language_groups(version=4.0):
    varieties = {'Portugese': ['pt-BR', 'pt-PT'],
                 'Spanish': ['es-ES', 'es-AR', 'es-PE'],
                 'English':['en-UK', 'en-US'],
                 'French': ['fr-CA', 'fr-FR'],
                 'Farsi': ['fa-AF', 'fa-IR']}
    
    similar_languages = {'Malay, Indo':['my', 'id'],
                         'Czech, Slovak':['cz', 'sk'],
                         'Bosnian, Croatian, Serbian':['bs', 'hr', 'sr'],
                         'Bulgarian, Macedonian':['bg','mk']}
                         
    if version == 1:
        similar_languages.pop('Bulgarian, Macedonian')
        varieties.pop('French')
        varieties.pop('Farsi')
    elif version == 4:
        varieties.pop('English')
        similar_languages.pop('Czech, Slovak')
        similar_languages.pop('Bulgarian, Macedonian')
        
    groups = varieties; groups.update(similar_languages)
    return groups
    

def breakdown_evaluation(results, goldtags, version=4.0, human_readable=True,
                         overall_only=False):    
    positives = Counter([y for x,y in zip(results,goldtags) if x.lower().replace('_', '-')==y.lower()])
    golds = Counter(goldtags)    
    
    groups = language_groups(version)

    sum_positives = sum(positives.values())
    sum_gold = sum(golds.values())
    accuracy = sum_positives / float(sum_gold)
    
    if overall_only:
        output_line = map(str, ['Overall', 'Accuracy', 
                                    sum_positives, sum_gold, accuracy])
        return accuracy
    
    bygroup_output = [] 
    
    if human_readable:
        print("=== Results === ")
        for g in groups:
            print()
            print(g)
            all_p = 0
            all_gl = 0
            for l in groups[g]:
                p = positives[l]
                gl = golds[l]
                all_p += p
                all_gl += gl
                print("{}: {} / {} = {}".format(l, p, gl, p/float(gl)))
            print("{}:  {} / {} = {}".format(g, all_p, all_gl, all_p*1. / all_gl))
        print("\nOverall:", "{} / {} = {}\n".format(sum_positives, sum_gold, accuracy))
    else:
        for g in groups:
            for l in groups[g]:
                p, gl = positives[l], golds[l]
                if not overall_only:
                    result_line = '\t'.join(map(str,[g, l, p, gl, p/float(gl)]))
                    bygroup_output.append(result_line)
        # print '\t'.join(output_line)
        return bygroup_output
        

def main(system_output, goldfile):
    results = [i.strip().split('\t')[-1] for i in io.open(system_output, 'r')]
    goldtags = [i.strip().split('\t')[-1] for i in io.open(goldfile, 'r')]
    breakdown_evaluation(results, goldtags, version=2.0, human_readable=True)
    
if __name__ == '__main__':
    if len(sys.argv) != 3:
        usage_msg = err_msg('Usage: python3 %s system_output goldfile\n'
                            % sys.argv[0])
        example_msg = err_msg('Example: python3 %s ~/usaar-dislang-run1-test.txt '
                              '~/DSLCC-v2.0/gold/test-gold.txt \n' % sys.argv[0])
        sys.stderr.write(usage_msg)
        sys.stderr.write(example_msg)
        sys.exit(1)

    main(*sys.argv[1:])