// dump_training_functions — MRA-NN training data generator
//
// Computes and exports to HDF5 the three functions needed for training:
//   rho0.mad.h5   — promolecular density (atomic superposition, NO SCF needed)
//   vnuc.mad.h5   — nuclear attraction potential (NO SCF needed)
//   rho.mad.h5    — converged SCF electron density (requires --archive)
//
// The first two can be generated from a plain moldft input file without running
// any SCF. The converged density is optional and requires a moldft restart archive.
//
// Usage:
//   dump_training_functions --input=<mol.in> [--archive=<prefix>.restartdata]
//                           [--out=DIR] [--thresh=X] [--k=N]
//
// All MRA parameters (k, thresh, L) are read from the input file and can be
// overridden on the command line. Output defaults to ./training_data/.
//
// HDF5 output requires a single MPI rank (--coeffs writer is single-rank).
// For parallel jobs, run moldft with many ranks but this tool with NP=1.

#include "../GroundState.hpp"
#include "../ResponseProtocol.hpp"

#include <madness/chem/SCF.h>
#include <madness/chem/molecule.h>
#include <madness/chem/molecular_functors.h>
#include <madness/misc/info.h>
#include <madness/mra/mra.h>
#include <madness/world/MADworld.h>

#ifdef MADNESS_HAS_HDF5
#include "../solvers/function_hdf5_io.hpp"
#endif

#include <filesystem>
#include <string>

using namespace madness;
using namespace molresponse_v3;

namespace {

// Save a real_function_3d to HDF5. Reconstructs first (writer needs leaf-only
// s-coefficients). Skips cleanly when NP>1 or HDF5 not compiled in.
void write_hdf5(World& world, real_function_3d f, const std::string& path) {
#ifdef MADNESS_HAS_HDF5
    if (world.size() > 1) {
        if (world.rank() == 0)
            print("  [HDF5] NP>1 — skipping", path,
                  "(run with a single rank for HDF5 output)");
        return;
    }
    f.reconstruct();
    molresponse_v3::save_function_hdf5(f, path);
    if (world.rank() == 0)
        print("  wrote", path);
#else
    if (world.rank() == 0)
        print("  [HDF5] not compiled — skipping", path);
#endif
}

}  // namespace

int main(int argc, char** argv) {
    World& world = initialize(argc, argv);
    int rc = 0;
    try {
        startup(world, argc, argv, true);

        commandlineparser parser(argc, argv);

        if (!parser.key_exists("input")) {
            if (world.rank() == 0) {
                print("Usage: dump_training_functions --input=<mol.in>");
                print("           [--archive=<prefix>.restartdata]");
                print("           [--out=DIR]  (default: training_data)");
                print("           [--thresh=X] [--k=N]");
                print("");
                print("  Exports rho0.mad.h5 (promolecular) and vnuc.mad.h5");
                print("  without running any SCF.  Add --archive to also");
                print("  export the converged rho.mad.h5.");
                print("  Requires a single MPI rank for HDF5 output.");
            }
            finalize();
            return 1;
        }

        {
            const std::string out_dir = parser.key_exists("out")
                ? parser.value_raw("out") : std::string("training_data");

            // ------------------------------------------------------------------
            // 1. Build SCF object from input file.
            //    The constructor reads the molecule, basis, and MRA parameters
            //    (k, thresh, L) and sets FunctionDefaults<3>::set_cubic_cell.
            // ------------------------------------------------------------------
            SCF calc(world, parser);

            // Override k / thresh from command line if provided.
            const int k = parser.key_exists("k")
                ? std::stoi(parser.value("k")) : calc.param.k();
            const double thresh = parser.key_exists("thresh")
                ? std::stod(parser.value("thresh")) : calc.param.econv();
            const double L = calc.param.L();

            FunctionDefaults<3>::set_k(k);
            FunctionDefaults<3>::set_thresh(thresh);
            FunctionDefaults<3>::set_cubic_cell(-L, L);
            FunctionDefaults<3>::set_refine(true);
            FunctionDefaults<3>::set_initial_level(2);
            FunctionDefaults<3>::set_truncate_mode(1);

            if (world.rank() == 0) {
                print("MRA-NN training data generator");
                print("  input  :", parser.value_raw("input"));
                print("  out    :", out_dir);
                print("  k      :", k);
                print("  thresh :", thresh);
                print("  L      :", L);
                print("  atoms  :", calc.molecule.natom());
            }

            if (world.rank() == 0)
                std::filesystem::create_directories(out_dir);
            world.gop.fence();

            // ------------------------------------------------------------------
            // 2. Nuclear potential V_nuc — no SCF needed.
            //    PotentialManager projects the Coulomb potential of all nuclei
            //    onto the MRA grid.
            // ------------------------------------------------------------------
            if (world.rank() == 0) print("\nComputing V_nuc ...");
            calc.make_nuclear_potential(world);
            real_function_3d vnuc = calc.potentialmanager->vnuclear();
            vnuc.truncate();
            write_hdf5(world, vnuc, out_dir + "/vnuc.mad.h5");

            // ------------------------------------------------------------------
            // 3. Promolecular density rho0 — no SCF needed.
            //    Superposition of spherically-averaged atomic densities from the
            //    STO-3G basis (same basis the SCF uses for its initial guess).
            // ------------------------------------------------------------------
            if (world.rank() == 0) print("\nComputing rho0 (promolecular) ...");
            functorT rho0_functor(
                new madchem::MolecularGuessDensityFunctor(calc.molecule, calc.aobasis));
            real_function_3d rho0 =
                real_factory_3d(world)
                    .functor(rho0_functor)
                    .truncate_on_project();
            rho0.truncate();
            write_hdf5(world, rho0, out_dir + "/rho0.mad.h5");

            if (world.rank() == 0) {
                print("  rho0 norm (integral) :", rho0.trace());
            }

            // ------------------------------------------------------------------
            // 4. Converged SCF density rho — requires --archive.
            //    Loads orbitals from a moldft restart and computes
            //    rho = sum_i occ_i * |psi_i|^2.
            // ------------------------------------------------------------------
            if (parser.key_exists("archive")) {
                const std::string archive_path = parser.value_raw("archive");
                if (world.rank() == 0)
                    print("\nLoading converged density from", archive_path, "...");

                auto gs = GroundState::from_archive(world, archive_path, calc.molecule);

                // rho = alpha density + beta density
                // For spin-restricted: beta = alpha, so rho = 2 * alpha
                real_function_3d rho = gs.scf().make_density(world,
                    gs.occupations_alpha(), gs.orbitals_alpha());
                if (gs.is_spin_restricted()) {
                    rho.scale(2.0);
                } else {
                    rho.gaxpy(1.0,
                        gs.scf().make_density(world,
                            gs.occupations_beta(), gs.orbitals_beta()),
                        1.0);
                }
                rho.truncate();

                if (world.rank() == 0)
                    print("  rho norm (electron count) :", rho.trace());

                write_hdf5(world, rho, out_dir + "/rho.mad.h5");
            }

            if (world.rank() == 0)
                print("\nDone. Output in:", out_dir);
        }

        finalize();
        return rc;
    } catch (const std::exception& e) {
        if (world.rank() == 0) print("Error:", e.what());
        finalize();
        return 1;
    }
}
