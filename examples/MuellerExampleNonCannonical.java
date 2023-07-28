import ch.idsia.credici.IO;
import ch.idsia.credici.factor.EquationBuilder;
import ch.idsia.credici.inference.CausalMultiVE;
import ch.idsia.credici.model.StructuralCausalModel;
import ch.idsia.credici.model.builder.EMCredalBuilder;
import ch.idsia.credici.model.transform.Cofounding;
import ch.idsia.credici.utility.DataUtil;
import ch.idsia.credici.utility.FactorUtil;
import ch.idsia.credici.utility.apps.SelectionBias;
import ch.idsia.crema.factor.bayesian.BayesianFactor;
import ch.idsia.crema.factor.credal.vertex.VertexFactor;
import ch.idsia.crema.utility.RandomUtil;
import com.opencsv.exceptions.CsvException;
import gnu.trove.map.TIntIntMap;
import jdk.jshell.spi.ExecutionControl;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;

/**
 * Example of a non-canonical SCM where the bounds are
 * tighte
 */

public class MuellerExampleNonCannonical {

    //// Numeric values identifying the variables and states in the model
    //
    static int T = 0;  //  Treatment
    static int S = 1;  // Survival
    static int G = 2;  // Gender
    public static void main(String[] args) throws InterruptedException, ExecutionControl.NotImplementedException, IOException, CsvException {



        // states for G, T and S
        int female=1, drug=1, survived=1;
        int male=0, no_drug=0, dead=0;

        // Relevant paths (update)
        String wdir = ".";
        String dataPath = Path.of(wdir, "./papers/journalPGM/models/literature/").toString();

        /* Example selection bias */

        // Load the data and the model
        TIntIntMap[] dataObs = DataUtil.fromCSV(Path.of(dataPath, "dataPearlObs.csv").toString());
        StructuralCausalModel model = (StructuralCausalModel) IO.readUAI(Path.of(dataPath, "consPearl.uai").toString());
        Cofounding.mergeExoParents(model, new int[][]{{T,S}});

        // Make the model non-canonical
        model = getReducedCo(model);


        BayesianFactor pXY = (BayesianFactor) DataUtil.getJointProb(dataObs, model.getDomain(T, S));
        double TPubound = pXY.filter(T,0).filter(S,0).getData()[0] + pXY.filter(T,1).filter(S,1).getData()[0];
        System.out.println("TianAndPearl result: [0,"+TPubound+"]");


        EMCredalBuilder builder = EMCredalBuilder.of(model, dataObs)
                .setNumTrajectories(100)
                .setWeightedEM(true)
                .setVerbose(false)
                .setMaxEMIter(200)
                .build();

        CausalMultiVE inf = new CausalMultiVE(builder.getSelectedPoints());
        VertexFactor resBiased = (VertexFactor) inf.probNecessityAndSufficiency(T, S, drug, no_drug, survived, dead);
        System.out.println("EMCC:"+ resBiased);

    }


    @NotNull
    private static StructuralCausalModel getReducedCo(StructuralCausalModel model) {
        //Start from the non-conservative model
        StructuralCausalModel m_reduced = model.copy();

        int U = m_reduced.getExogenousParents(T)[0];



        // Modify the SEs and exogenous domains

        m_reduced.removeVariable(U);
        m_reduced.addVariable(U,7,true);
        m_reduced.addParents(T,U);
        m_reduced.addParents(S,U,T);

        RandomUtil.setRandomSeed(0);
        m_reduced.fillWithRandomEquations();


        return m_reduced;
    }

}
