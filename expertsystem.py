from experta import *
from textinterface import ask_question

def runex():
    engine = DiagnosticsES()
    engine.reset()
    engine.run()

class DiagnosticsES(KnowledgeEngine):

    @DefFacts()
    def _initial_action(self):
        yield Fact(question=True)

    # SINTOMI eguali
    @Rule(Fact(question=True))
    def ask_mal_di_testa(self):
        self.declare(Fact(mal_di_testa=ask_question("Hai mal di testa ?")))

    @Rule(Fact(question=True))
    def ask_mal_di_gola(self):
        self.declare(Fact(mal_di_gola=ask_question("Hai mal di gola ?")))
        
    @Rule(Fact(question=True))
    def ask_dolori_muscolari(self):
        self.declare(Fact(dolori_muscolari=ask_question("Hai dolori muscolari ?")))

    @Rule(OR(Fact(mal_di_testa=True), Fact(mal_di_gola=True), Fact(dolori_muscolari=True)))
    def sintomi_eguali(self):
        self.declare(Fact(sintomi_eguali=True))

    @Rule(AND(Fact(mal_di_testa=False), Fact(mal_di_gola=False), Fact(dolori_muscolari=False)))
    def no_sintomi_eguali(self):
        self.declare(Fact(sintomi_eguali=False))

    # febbre
    @Rule(Fact(sintomi_eguali=True))
    def ask_febbre(self):
        self.declare(Fact(febbre=ask_question("Hai la febbre?")))

    # temperatura alta
    @Rule(OR(Fact(febbre=True)))
    def ask_temperatura_alta(self):
        self.declare(Fact(temperatura_alta=ask_question("Hai la temperatura alta?")))

    # esame
    @Rule(Fact(temperatura_alta=True))
    def ask_esame(self):
        self.declare(Fact(esame_fatto=ask_question("Hai svolto un RM ai polmoni per vedere se hai qualcosa?")))

    @Rule(Fact(esame_fatto=True))
    def ask_positivo(self):
        self.declare(Fact(esame=ask_question("Hanno trovato qualcosa di grave?")))

    # dolore al petto
    @Rule(Fact(esame_fatto=False))
    def ask_dolore_al_petto(self):
        self.declare(Fact(dolore_al_petto=ask_question("Hai dolore al petto?")))

    # difficoltà di concentrazione
    @Rule(OR(Fact(esame=False), Fact(dolore_al_petto=False), Fact(temperatura_alta=False), Fact(febbre=False)))
    def ask_difficoltà_di_concentrazione(self):
        self.declare(Fact(difficoltà_di_concentrazione=ask_question("Hai difficoltà a concentrarti?")))

    # tosse
    @Rule(OR(Fact(dolore_al_petto=True), Fact(difficoltà_di_concentrazione=True)))
    def ask_tosse(self):
        self.declare(Fact(tosse=ask_question("Hai tosse secca?")))

    # rm_polmoni
    @Rule(OR(AND(Fact(tosse=True), Fact(dolore_al_petto=True)), Fact(esame=True)))
    def rm_polmoni(self):
        self.declare(Fact(rm_polmoni=True))

    # stanchezza
    @Rule(OR(AND(Fact(tosse=True), Fact(difficoltà_di_concentrazione=True), Fact(febbre=True))))
    def stanchezza(self):
        self.declare(Fact(stanchezza=True))

    # covid
    @Rule(AND(Fact(sintomi_eguali=True), Fact(temperatura_alta=True), Fact(rm_polmoni=True)))
    def covid(self):
        print("I sintomi indicano che potresti aver contratto il virus del covid.")
        print("È consiglibile recarsi in una struttura ospedaliera il prima possibile.")
        self.reset()

    # influenza
    @Rule(AND(Fact(sintomi_eguali=True), Fact(stanchezza=True)))
    def influenza(self):
        print("I sintomi indicano che potresti avere un'influenza.")
        print("È consigliabile recarsi da un medico per ulteriori accertamenti o terapie.")
        self.reset()

    # malattia assente
    @Rule(OR(Fact(sintomi_eguali=False), Fact(tosse=False),
          OR(Fact(difficoltà_di_concentrazione=False),
          OR(Fact(febbre=False)),
          AND(Fact(dolore_al_petto=False), Fact(difficoltà_di_concentrazione=False)),
          AND(Fact(difficoltà_di_concentrazione=False), Fact(temperatura_alta=False)),
          AND(Fact(difficoltà_di_concentrazione=False), Fact(tosse=True)))))
    def malattiaassente(self):
        print("Con i sintomi indicati non dovresti aver nessun virus.")
        self.reset()
