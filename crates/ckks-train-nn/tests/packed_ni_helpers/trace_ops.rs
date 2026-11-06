//! Private CPU operand witness. Matches every RNS coefficient, including lowered prefixes.
use super::*;
pub struct TraceOps {
    pub ev: Evaluator,
    pub sentinel: Ciphertext,
    pub affine_uses: Cell<usize>,
    pub outer_uses: Cell<usize>,
}
fn prefix(a: &Ciphertext, b: &Ciphertext) -> bool {
    a.len() == b.len()
        && a.level() <= b.level()
        && a.scale().to_bits() == b.scale().to_bits()
        && a.polys()
            .iter()
            .zip(b.polys())
            .all(|(x, y)| x.limbs() == &y.limbs()[..=a.level()])
}
macro_rules! delegate {
 ($name:ident($($a:ident:$t:ty),*)->$r:ty) => {fn $name(&self,$($a:$t),*)->$r {HeOps::$name(&self.ev,$($a),*)}};
}
impl HeOps for TraceOps {
    type Ct = Ciphertext;
    type Pt = ckks_train_ckks::Plaintext;
    delegate!(params()->&Arc<CkksParams>);
    delegate!(config()->&ckks_train_ckks::EvalConfig);
    delegate!(counter()->OpCounter);
    delegate!(reset_counter()->());
    delegate!(add_counts(counts:&OpCounter)->());
    fn fork(&self) -> Self {
        Self {
            ev: HeOps::fork(&self.ev),
            sentinel: self.sentinel.clone(),
            affine_uses: Cell::new(0),
            outer_uses: Cell::new(0),
        }
    }
    delegate!(has_relin_key()->bool);
    delegate!(has_galois_key(step:i32)->bool);
    delegate!(owns(ct:&Ciphertext)->bool);
    delegate!(upload_ct(ct:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(download_ct(ct:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(upload_pt(pt:&ckks_train_ckks::Plaintext)->ckks_train_core::error::Result<ckks_train_ckks::Plaintext>);
    delegate!(add(a:&Ciphertext,b:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(sub(a:&Ciphertext,b:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(negate(a:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(add_plain(a:&Ciphertext,p:&ckks_train_ckks::Plaintext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(mul_plain(a:&Ciphertext,p:&ckks_train_ckks::Plaintext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(mul_scalar(ct:&Ciphertext,value:f64,target_scale:f64)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(mul_plain_to_scale(a:&Ciphertext,p:&ckks_train_ckks::Plaintext,target_scale:f64)->ckks_train_core::error::Result<Ciphertext>);
    fn mul_no_relin(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        if prefix(a, &self.sentinel) || prefix(b, &self.sentinel) {
            self.outer_uses.set(self.outer_uses.get() + 1);
        }
        self.ev.mul_no_relin(a, b)
    }
    delegate!(relinearize(ct:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    fn mul(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        if prefix(a, &self.sentinel) || prefix(b, &self.sentinel) {
            self.affine_uses.set(self.affine_uses.get() + 1);
        }
        self.ev.mul(a, b)
    }
    delegate!(rescale(ct:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(mod_switch_to_level(ct:&Ciphertext,level:usize)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(relabel(ct:Ciphertext,scale:f64)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(rotate(ct:&Ciphertext,steps:i32)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(conjugate(ct:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(hoisted_rotate(ct:&Ciphertext,steps:&[i32])->ckks_train_core::error::Result<Vec<Ciphertext>>);
    delegate!(matvec_plain(diag:&ckks_train_ckks::Diagonals<ckks_train_ckks::Plaintext>,x:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(matvec_enc(diag:&ckks_train_ckks::Diagonals<Ciphertext>,x:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
}
