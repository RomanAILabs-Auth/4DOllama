package quantum

// Hamiltonian model (discrete Trotter, product-state truncation on bonds).
//
// Local generator per site (nq qubits, Dim = 2^nq):
//   H_local = Σ_k hz[k] Z_k + Σ_k hx[k] X_k
// Z_k commute among themselves; X_k commute among themselves; Z and X on same k do not.
// Strang splitting on each site (exact unitary on C^Dim):
//   U_loc = exp(-i (Δt/2) H_Z) exp(-i Δt H_X) exp(-i (Δt/2) H_Z)
//   H_Z = Σ_k hz[k] Z_k,  H_X = Σ_k hx[k] X_k.
//
// Bond generator between sites i and j (same k on each side):
//   H_ij = J Σ_{k=0}^{nq-1} X_{i,k} X_{j,k}
// The summands commute (disjoint Pauli supports on the 2*nq-qubit tensor space).
//   U_ij = Π_k exp(-i Δt J X_{i,k} X_{j,k})
// Each factor acts on the bond tensor η ∈ C^{Dim}⊗C^{Dim} encoded as length Dim² with
// layout idx = a·Dim + b for local indices a,b.
//
// Product-state storage: the global ansatz is ⊗_s |ψ_s⟩. The exact image U_ij (ψ_i⊗ψ_j) is
// generally entangled. This simulator projects it onto the closest rank-one tensor in
// Frobenius norm (leading Schmidt pair), using fixed-iteration Hermitian power iteration
// with stack scratch only. That projection is not an exact many-body unitary on (C^Dim)^{⊗N}.
//
// After each bond, the two affected sites are rescaled jointly so that
//   ‖ψ_i‖² + ‖ψ_j‖²
// matches its pre-bond value, preserving the global quantity Σ_s ‖ψ_s‖² (concatenated ℓ²
// norm over all sites). This is explicit documented norm control, not an exact unitary on
// the truncated manifold.

// SetUniformFields sets hz[k] = hz0 and hx[k] = hx0 for all local qubits.
func (S *Simulator) SetUniformFields(hz0, hx0 float64) {
	for k := 0; k < S.NQ; k++ {
		S.Hz[k] = hz0
		S.Hx[k] = hx0
	}
}
