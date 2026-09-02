class Thesaurus():
    def __init__(self):
        self.words = dict()
        self.add_homophones('cozmo', \
                            ["cozimo","cosimo","cosmo", \
                             "kozmo","cosmos","cozmos"])
        self.add_homophones('right', ['write','wright'])
        self.add_homophones('1',['one','won'])
        self.add_homophones('cube1',['q1','coupon','cuban'])
        self.phrase_tree = dict()
        self.add_phrases('cube1',['cube 1'])
        self.add_phrases('cube2',['cube 2'])
        self.add_phrases('cube2',['cube to'])
        self.add_phrases('cube3',['cube 3'])
        self.add_phrases('paperclip',['paper clip'])
        self.add_phrases('deli-slicer',['deli slicer'])
        

    def add_homophones(self,word,homophones):
        if not isinstance(homophones,list):
            homophones = [homophones]
        for h in homophones:
            self.words[h] = word

    def lookup_word(self,word):
        return self.words.get(word,word)

    def add_phrases(self,word,phrases):
        if not isinstance(phrases,list):
            phrases = [phrases]
        for phrase in phrases:
            wdict = self.phrase_tree
            for pword in phrase.split(' '):
                wdict[pword] = wdict.get(pword,dict())
                wdict = wdict[pword]
            wdict[''] = word

    def substitute_phrases(self,words):
        result = []
        while words != []:
            word = words[0]
            del words[0]
            wdict = self.phrase_tree.get(word,None) 
            if wdict is None:
                result.append(word)
                continue
            prefix = [word]
            while words != []:
                wdict2 = wdict.get(words[0],None)
                if wdict2 is None: break
                prefix.append(words[0])
                del words[0]
                wdict = wdict2
            subst = wdict.get('',None)
            if subst is not None:
              result.append(subst)
            else:
              result = result + prefix
        return result
