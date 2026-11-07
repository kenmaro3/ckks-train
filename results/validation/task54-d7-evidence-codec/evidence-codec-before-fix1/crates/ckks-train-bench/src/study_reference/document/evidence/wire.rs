//! Borrowed fixed-depth primitives. All allocation counts preflight remaining bytes.
pub(in crate::study_reference::document) use super::super::execution_descriptors::Cursor;
use super::*;
pub(in crate::study_reference::document) trait Wire<'a>:
    Sized
{
    const MIN: usize;
    fn read(c: &mut Cursor<'a>) -> Result<Self>;
    fn write(&self, e: &mut Encoder) -> Result<()>;
}
impl<'a> Wire<'a> for u64 {
    const MIN: usize = 8;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        c.u64()
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        e.u64(*self)
    }
}
impl<'a> Wire<'a> for i32 {
    const MIN: usize = 4;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        let mut b = [0; 4];
        b.copy_from_slice(c.take(4)?);
        Ok(Self::from_le_bytes(b))
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        e.append(&self.to_le_bytes(), false)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::study_reference::document) struct Tag<const M: u8>(
    pub(in crate::study_reference::document) u8,
);
impl<'a, const M: u8> Wire<'a> for Tag<M> {
    const MIN: usize = 1;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        c.tag(M).map(Self)
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        ensure(self.0 <= M, DocumentErrorCode::UnsupportedTag, "tag")?;
        e.byte(self.0)
    }
}
impl<'a, T: Wire<'a>> Wire<'a> for Option<T> {
    const MIN: usize = 1;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        c.option(T::read)
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        e.byte(u8::from(self.is_some()))?;
        if let Some(v) = self {
            v.write(e)?;
        }
        Ok(())
    }
}
pub(super) fn read_vec<'a, T: Wire<'a>>(c: &mut Cursor<'a>, max: usize) -> Result<Vec<T>> {
    let n = c.bounded_count(max, T::MIN)?;
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| err(DocumentErrorCode::TooLarge, "list.allocation"))?;
    for _ in 0..n {
        out.push(T::read(c)?);
    }
    Ok(out)
}
impl<'a, T: Wire<'a>> Wire<'a> for Vec<T> {
    const MIN: usize = 8;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        read_vec(c, 65_536)
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        ensure(
            self.len() <= 65_536,
            DocumentErrorCode::TooLarge,
            "list.count",
        )?;
        e.u64(self.len() as u64)?;
        for v in self {
            v.write(e)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::study_reference::document) struct Str<'a, const N: usize>(
    pub(in crate::study_reference::document) &'a str,
);
impl<'a, const N: usize> Wire<'a> for Str<'a, N> {
    const MIN: usize = 8;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        c.bounded_text(N).map(Self)
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        ensure(self.0.len() <= N, DocumentErrorCode::TooLarge, "text")?;
        e.text(self.0)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::study_reference::document) struct Id<'a>(
    pub(in crate::study_reference::document) &'a str,
);
impl<'a> Wire<'a> for Id<'a> {
    const MIN: usize = 9;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        let s = c.bounded_text(128)?;
        target::valid_id(s, "identifier")?;
        Ok(Self(s))
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        target::valid_id(self.0, "identifier")?;
        e.text(self.0)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::study_reference::document) struct H<'a>(
    pub(in crate::study_reference::document) &'a str,
);
impl<'a> Wire<'a> for H<'a> {
    const MIN: usize = 72;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        let s = c.bounded_text(64)?;
        ensure(
            s.len() == 64 && lower_hex(s),
            DocumentErrorCode::BadHex,
            "hash",
        )?;
        Ok(Self(s))
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        ensure(
            self.0.len() == 64 && lower_hex(self.0),
            DocumentErrorCode::BadHex,
            "hash",
        )?;
        e.text(self.0)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::study_reference::document) struct Git<'a>(
    pub(in crate::study_reference::document) &'a str,
);
impl<'a> Wire<'a> for Git<'a> {
    const MIN: usize = 48;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        let s = c.bounded_text(40)?;
        ensure(
            s.len() == 40 && lower_hex(s),
            DocumentErrorCode::BadHex,
            "revision",
        )?;
        Ok(Self(s))
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        ensure(
            self.0.len() == 40 && lower_hex(self.0),
            DocumentErrorCode::BadHex,
            "revision",
        )?;
        e.text(self.0)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::study_reference::document) struct Bytes<'a>(
    pub(in crate::study_reference::document) &'a [u8],
);
impl<'a> Wire<'a> for Bytes<'a> {
    const MIN: usize = 8;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        c.bounded_bytes(A_MAX).map(Self)
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        ensure(self.0.len() <= A_MAX, DocumentErrorCode::TooLarge, "bytes")?;
        e.u64(self.0.len() as u64)?;
        e.append(self.0, false)
    }
}
// These existing frozen records share their existing output traversal.
impl<'a> Wire<'a> for Link {
    const MIN: usize = 144;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        Ok(Self {
            file_sha256: Hash(H::read(c)?.0.into()),
            payload_sha256: Hash(H::read(c)?.0.into()),
        })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.encode(e)
    }
}
impl<'a> Wire<'a> for Clock {
    const MIN: usize = 18;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        c.tag(0)?;
        let unix_ns = c.u64()?;
        c.tag(0)?;
        let elapsed_ns = c.u64()?;
        Ok(Self {
            wall_source: WallSource::Unix,
            unix_ns,
            monotonic_source: MonotonicSource::Instant,
            elapsed_since_operation_start_ns: elapsed_ns,
        })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.encode(e)
    }
}
impl<'a> Wire<'a> for admission::Predecessor {
    const MIN: usize = 154;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        Ok(Self {
            attempt_id: Text(Id::read(c)?.0.into()),
            admission: Link::read(c)?,
            start: Option::read(c)?,
        })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.encode(e)
    }
}
impl<'a> Wire<'a> for admission::Gates {
    const MIN: usize = 5 * 81;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        fn gate(c: &mut Cursor<'_>) -> Result<admission::Gate> {
            Ok(admission::Gate {
                policy_revision: Text(Id::read(c)?.0.into()),
                evidence_sha256: Hash(H::read(c)?.0.into()),
            })
        }
        Ok(Self {
            inputs: gate(c)?,
            implementation: gate(c)?,
            layout: gate(c)?,
            resource: gate(c)?,
            user: gate(c)?,
        })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.encode(e)
    }
}
macro_rules! wire_record {
    ($name:ident { $($f:ident:$t:ty),* $(,)? }) => {
        #[derive(Debug,Clone)]
        pub(in crate::study_reference::document) struct $name<'a> {$(pub(in crate::study_reference::document) $f:$t,)*}
        impl<'a> Wire<'a> for $name<'a> {
            const MIN:usize=0 $(+ <$t as Wire<'a>>::MIN)*;
            fn read(c:&mut Cursor<'a>)->Result<Self>{Ok(Self{$($f:<$t as Wire<'a>>::read(c)?,)*})}
            fn write(&self,e:&mut Encoder)->Result<()> {$ (self.$f.write(e)?;)*Ok(())}
        }
    };
}
pub(in crate::study_reference::document) use wire_record;
