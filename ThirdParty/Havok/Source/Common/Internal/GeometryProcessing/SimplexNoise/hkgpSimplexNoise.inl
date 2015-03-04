/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

namespace	hkgpSimplexNoise
{
	// Jenkins hash.
	HK_FORCE_INLINE hkUint32	hash(hkUint32 x)
	{
		x = (x+0x7ed55d16) + (x<<12);
		x = (x^0xc761c23c) ^ (x>>19);
		x = (x+0x165667b1) + (x<<5);
		x = (x+0xd3a2646c) ^ (x<<9);
		x = (x+0xfd7046c5) + (x<<3);
		x = (x^0xb55a4f09) ^ (x>>16);
		return x;
	}
	
	//
	template <int N>
	HK_FORCE_INLINE hkUint32	hash(const hkUint32* x)
	{
		#if 0
		struct Functor
		{
			HK_FORCE_INLINE Functor(const hkUint32* y) : m_y(y) {}
			HK_FORCE_INLINE hkUint32 operator()(int i) const { return m_y[i]; }
			const hkUint32*	m_y;
		}	functor(x);

		hkUint32	part0 = 0;
		hkUint32	part1 = 0;

		hashT<N-1>::unroll(functor,part0,part1);
		
		return hash(part0 + part1);
		#else
		hkUint32	h = 0;
		for(int i=0; i<N; ++i)
		{
			h ^= x[i] << (i * 8);
		}
		return hash(h);
		#endif
	}

	//
	template <int INDEX>
	struct hashT
	{
		template <typename T>
		static HK_FORCE_INLINE void unroll(T& functor, hkUint32& p0, hkUint32& p1)
		{
			hashT<INDEX-1>::unroll(functor,p0,p1);

			const hkUint32	x = functor(INDEX);
			p0 ^= x + hkUint32((hkUint64(INDEX) * 2654435789ul) & 0xfffffffful);
			p1 ^= hash(x);			
		}
	};
	
	//
	template <> struct hashT<-1> { template <typename T> static HK_FORCE_INLINE void unroll(T&, hkUint32&, hkUint32&) {} };	

	//
	template <int N> struct Pow2 { enum { VALUE = 2 * Pow2<N-1>::VALUE }; };

	//
	template <> struct Pow2<0> { enum { VALUE = 1 }; };

	//
	template <int INDEX>
	struct ComputeOrigin
	{
		static HK_FORCE_INLINE void unroll(const hkReal* p, int* o, hkReal s)
		{
			ComputeOrigin<INDEX-1>::unroll(p,o,s);

			const hkReal	x = p[INDEX] + s;
			o[INDEX]	=	int(x) - (hkMath::signBitSet(x) ? 1 : 0);
		}
	};

	//
	template <> struct ComputeOrigin<-1> { static HK_FORCE_INLINE void unroll(const hkReal*, int*, hkReal) {} };

	//
	template <int INDEX>
	struct ComputeCornerAndUnskewFactor
	{
		template <int CI>
		static HK_FORCE_INLINE void unroll(const int* o, int* c, hkReal& uso)
		{
			ComputeCornerAndUnskewFactor<INDEX-1>::unroll<CI>(o,c,uso);

			c[INDEX]	=	o[INDEX] + ((CI >> INDEX) & 1);
			uso			+=	c[INDEX];
		}
	};

	//
	template <> struct ComputeCornerAndUnskewFactor<-1> { template <int CI> static HK_FORCE_INLINE void unroll(const int*, int*, hkReal&) {} };

	//
	template <int INDEX>
	struct ComputeFractionsAndK
	{
		static HK_FORCE_INLINE void unroll(const hkReal* p, const int* c, hkReal uso, hkReal* f, hkReal& k)
		{
			ComputeFractionsAndK<INDEX-1>::unroll(p,c,uso,f,k);

			f[INDEX]	=	p[INDEX] - (c[INDEX] - uso);
			k			-=	f[INDEX] * f[INDEX];
		}
	};

	//
	template <> struct ComputeFractionsAndK<-1> { static HK_FORCE_INLINE void unroll(const hkReal*, const int*, hkReal, hkReal*, hkReal&) {} };

	//
	template <int INDEX>
	struct ComputeGradients
	{
		template <int BITS_PER_AXIS, bool STORE_GRADIENTS>
		static HK_FORCE_INLINE void unroll(const hkReal* f, hkUint32 h, hkReal nf, hkReal* g, hkReal k, hkReal& d)
		{
			ComputeGradients<INDEX-1>::unroll<BITS_PER_AXIS,STORE_GRADIENTS>(f,h,nf,g,k,d);

			const hkUint32	mask = (1 << BITS_PER_AXIS) - 1;
			const hkReal	mh = hkReal((h >> (INDEX * BITS_PER_AXIS)) & mask);
			const hkReal	a = (mh - mask / hkReal(2)) * nf;
			if(STORE_GRADIENTS) g[INDEX] = a * k;
			d += f[INDEX] * a;
		}
	};

	//
	template <> struct ComputeGradients<-1> { template <int BITS_PER_AXIS, bool STORE_GRADIENTS> static HK_FORCE_INLINE void unroll(const hkReal*, hkUint32, hkReal, hkReal*, hkReal, hkReal&) {} };

	//
	template <int INDEX>
	struct ComputeFirstDerivative
	{
		static HK_FORCE_INLINE void unroll(hkReal factor, hkReal d8, const hkReal* g, const hkReal* f, hkReal* dOut)
		{
			ComputeFirstDerivative<INDEX-1>::unroll(factor,d8,g,f,dOut-1);

			*dOut += factor * (g[INDEX] - d8 * f[INDEX]);
		}
	};

	//
	template <> struct ComputeFirstDerivative<-1> { static HK_FORCE_INLINE void unroll(hkReal, hkReal, const hkReal*, const hkReal*, hkReal*) {} };
	
	//
	template <int INDEX>
	struct ComputeCorner
	{
		template <int N, bool D1, bool D2>
		static HK_FORCE_INLINE void unroll(const hkReal* p, const int* o, hkReal uf, hkReal r, hkReal rInv, hkReal nf, hkReal dfb, hkReal& value, hkReal* der1, hkReal* der2)
		{
			ComputeCorner<INDEX-1>::template unroll<N,D1,D2>(p,o,uf,r,rInv,nf,dfb,value,der1,der2);

			int		c[N];
			hkReal	uso=0;
			ComputeCornerAndUnskewFactor<N-1>::unroll<INDEX>(o, c, uso);
			uso *= uf;
			
			hkReal		f[N];
			hkReal		k = r;
			ComputeFractionsAndK<N-1>::unroll(p, c, uso, f, k);
			
			if(k > 0)
			{
				hkReal		g[N];
				hkReal		d = 0;
				hkUint32	h = hash<N>((const hkUint32*)c);
				
				// Gradients.
				ComputeGradients<N-1>::unroll<32/(N+1),D1|D2>(f, h, nf, g, k, d);
				
				// Value.
				value	+=	hkMath::pow<4>(k * rInv) * d;

				// First derivative.
				if(D1)
				{
					const hkReal	factor = hkMath::pow<3>(k) * dfb;
					const hkReal	d8 = 8 * d;
					ComputeFirstDerivative<N-1>::unroll(factor,d8,g,f,der1 + N - 1);
				}

				// Second derivative.
				if(D2)
				{
					const hkReal	factor = -8 * hkMath::pow<2>(k) * dfb;
					const hkReal	d6 = 6 * d;
					hkReal*			dOut = der2;
					for(int i=0; i<N; ++i)
					{
						for(int j=i; j<N; ++j)
						{
							if(i == j)
								*dOut++ += factor * (2 * g[i] * f[i] + d * (k - 6 * f[i] * f[i]));
							else
								*dOut++ += factor * (g[j] * f[i] + g[i] * f[j] - d6 * f[i] * f[j]);
						}
					}
				}
			}
		}
	};

	//
	template <> struct ComputeCorner<-1> { template <int N, bool D1, bool D2> static HK_FORCE_INLINE void unroll(const hkReal*, const int*, hkReal, hkReal, hkReal, hkReal, hkReal, hkReal&, hkReal*, hkReal*) {} };
};

//
template <int N, bool FIRST_DER, bool SECOND_DER>
inline hkReal	hkgpSimplexNoise::evaluate(const hkReal* positions, hkReal* der1, hkReal* der2)
{
	HK_COMPILE_TIME_ASSERT(0 < N);
	// Contants.
	const int		bitsPerAxis = 32 / (N+1);
	const hkUint32	bitMask = (1 << bitsPerAxis) - 1;
	const hkReal	bitHalf = bitMask / 2.0f;
	const hkReal	bitHalfInv = 1 / bitHalf;
	const hkReal	denBase = hkMath::sqrt(hkReal(N+1)) + 1;
	const hkReal	skewFactor = 1 / denBase;
	const hkReal	unskewFactor = 1 / (N + denBase);
	const hkReal	radiusSquared = N / hkReal(1+N);
	const hkReal	radiusSquaredInv = hkReal(1+N) / N;
	const hkReal	normalizationFactor = hkReal(1.9f) * bitHalfInv * (hkReal(1+N) / N);
	const hkReal	derFactorBase = 1 / hkMath::pow<4>(radiusSquared);

	// Reset derivatives.
	if(FIRST_DER)	for(int i=0; i<N; ++i) der1[i] = 0;
	if(SECOND_DER)	for(int i=0; i<((N*(1+N))/2); ++i) der2[i] = 0;

	// Compute skew offset.
	hkReal	skewOffset = 0;
	for(int i=0; i<N; ++i) skewOffset += positions[i];
	skewOffset *= skewFactor;

	// Compute origin.
	int		origins[N];
	ComputeOrigin<N-1>::unroll(positions,origins,skewOffset);
		
	// Compute value and derivatives.
	hkReal	value = 0;
	#if 0	// Unroll all.
	ComputeCorner<Pow2<N>::VALUE - 1>::unroll<N,FIRST_DER,SECOND_DER>(positions, origins, unskewFactor, radiusSquared, radiusSquaredInv, normalizationFactor, derFactorBase, value, der1, der2);
	#else	// Loop.
	for(int cornerIndex = 0; cornerIndex < Pow2<N>::VALUE; ++cornerIndex)
	{
		int		c[N];
		hkReal	uso=0;
		for(int i=0; i<N; ++i)
		{
			c[i]	=	origins[i] + ((cornerIndex >> i) & 1);
			uso		+=	c[i];
		}
		uso *= unskewFactor;
			
		hkReal		f[N];
		hkReal		k = radiusSquared;			
		for(int i=0; i<N; ++i)
		{
			f[i]	=	positions[i] - (c[i] - uso);
			k		-=	f[i] * f[i];
		}
			
		if(k > 0)
		{
			hkReal		g[N];
			hkReal		d = 0;
			hkUint32	h = hash<N>((const hkUint32*)c);
				
			for(int i=0; i<N; ++i, h >>= bitsPerAxis)
			{
				const hkReal	a = (hkReal(h & bitMask) - bitHalf) * normalizationFactor;
				if(FIRST_DER | SECOND_DER) g[i] = a * k;
				d += f[i] * a;
			}

			// Value.
			value	+=	hkMath::pow<4>(k * radiusSquaredInv) * d;

			// First derivative.
			if(FIRST_DER)
			{
				const hkReal	factor = hkMath::pow<3>(k) * derFactorBase;
				const hkReal	d8 = 8 * d;
				hkReal*			dOut = der1;
				for(int i=0; i<N; ++i)
				{
					*dOut++ += factor * (g[i] - d8 * f[i]);
				}
			}

			// Second derivative.
			if(SECOND_DER)
			{
				const hkReal	factor = -8 * hkMath::pow<2>(k) * derFactorBase;
				const hkReal	d6 = 6 * d;
				hkReal*			dOut = der2;
				for(int i=0; i<N; ++i)
				{
					for(int j=0; j<=i; ++j)
					{
						if(i == j)
							*dOut++ += factor * (2 * g[i] * f[i] + d * (k - 6 * f[i] * f[i]));
						else
							*dOut++ += factor * (g[j] * f[i] + g[i] * f[j] - d6 * f[i] * f[j]);
					}
				}
			}
		}
	}
	#endif
	return value;
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
