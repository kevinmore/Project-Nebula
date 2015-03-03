/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Common/Base/Math/Vector/Mx/hkMxUnroll.h>

// mxvector.cpp test assumes 'precise' float control,
// and can fail if model is set to 'fast'. (See issue COM-1779)
#ifdef HK_PLATFORM_WIN32
	#if (HK_CONFIG_SIMD==HK_CONFIG_SIMD_DISABLED)
		#pragma float_control(precise, on)
		#pragma float_control(except, on)
	#endif
#endif

#if defined(HK_COMPILER_HAS_INTRINSICS_IA32)
	#if (HK_SSE_VERSION >= 0x50) && !defined(HK_REAL_IS_DOUBLE)
		#define MX_VECTOR_IS_AVX
	#endif
#endif

HK_ALIGN_REAL(static const hkReal s_mxvector_aligned_data[32]) = { 0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15,  0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15 };
static hkUint8 s_mxvector_byte_data[512];

template <int M>
struct _MXVector
{
    hkReal val[M*4];
};

template <int M>
static void mxvector_load_assign()
{
	{
		hkMxVector<M> v; v.moveLoad((hkVector4*)s_mxvector_aligned_data);
		const _MXVector<M>* c = (const _MXVector<M>*)&(v.m_vec);
		for (int i=0; i<M*4; ++i)
		{
			HK_TEST(c->val[i] == s_mxvector_aligned_data[i]);
		}
	}	
	{
		hkMxVector<M> v; v.template load<4>(s_mxvector_aligned_data);
		const _MXVector<M>* c = (const _MXVector<M>*)&(v.m_vec);
		for (int i=0; i<M*4; ++i)
		{
			HK_TEST(c->val[i] == s_mxvector_aligned_data[i]);
		}
	}	
	{
		hkMxVector<M> v; v.loadNotCached(s_mxvector_aligned_data);
		const _MXVector<M>* c = (const _MXVector<M>*)&(v.m_vec);
		for (int i=0; i<M*4; ++i)
		{
			HK_TEST(c->val[i] == s_mxvector_aligned_data[i]);
		}
	}	
#if 0
	// byte-aligned not supported on all platforms yet
	{
		const hkUint8* dataStart = s_mxvector_byte_data+3;
		hkMxVector<M> v; v.loadNotAligned((const hkReal*)dataStart);
		const hkUint8* c = (const hkUint8*)&(v.m_vec);
		for (int i=0; i<M*4*hkSizeOf(hkReal); ++i)
		{
			HK_TEST(c[i] == dataStart[i]);
		}
	}	
#endif

#ifndef MX_VECTOR_IS_AVX
	{
		// Not loaded components are undefined.

		hkMxVector<M> v; 
		const _MXVector<M>* c = (const _MXVector<M>*)&(v.m_vec);

		v.setZero(); v.template load<4*sizeof(hkReal), 1, HK_IO_NATIVE_ALIGNED>(s_mxvector_aligned_data);
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c->val[i+0] == s_mxvector_aligned_data[i+0]);
			//HK_TEST(c[i+1] == 0.0f); HK_TEST(c[i+2] == 0.0f); HK_TEST(c[i+3] == 0.0f);
		}

		v.setZero(); v.template load<4*sizeof(hkReal), 2, HK_IO_NATIVE_ALIGNED>(s_mxvector_aligned_data);
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c->val[i+0] == s_mxvector_aligned_data[i+0]); 
			HK_TEST(c->val[i+1] == s_mxvector_aligned_data[i+1]); 
			//HK_TEST(c[i+2] == 0.0f); HK_TEST(c[i+3] == 0.0f);
		}

		v.setZero(); v.template load<4*sizeof(hkReal), 3, HK_IO_NATIVE_ALIGNED>(s_mxvector_aligned_data);
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c->val[i+0] == s_mxvector_aligned_data[i+0]); 
			HK_TEST(c->val[i+1] == s_mxvector_aligned_data[i+1]); 
			HK_TEST(c->val[i+2] == s_mxvector_aligned_data[i+2]); 
			//HK_TEST(c[i+3] == 0.0f);
		}

		v.setZero(); v.template load<4*sizeof(hkReal), 4, HK_IO_NATIVE_ALIGNED>(s_mxvector_aligned_data);
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c->val[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c->val[i+1] == s_mxvector_aligned_data[i+1]);
			HK_TEST(c->val[i+2] == s_mxvector_aligned_data[i+2]);
			HK_TEST(c->val[i+3] == s_mxvector_aligned_data[i+3]); 
		}
	}
	{
		// Not loaded components are undefined.

		const hkReal* base[M]; for (int i=0; i<M; i++) base[i] = (hkReal*) (((char*)(s_mxvector_aligned_data+i*4))-3);
		hkMxVector<M> v; 
		const _MXVector<M>* cc = (const _MXVector<M>*)&(v.m_vec);

		v.setZero(); v.template loadWithOffset<3, 1, HK_IO_NATIVE_ALIGNED>(base);
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(cc->val[i+0] == s_mxvector_aligned_data[i+0]); 
			//HK_TEST(cc[i+1] == 0.0f); HK_TEST(cc[i+2] == 0.0f); HK_TEST(cc[i+3] == 0.0f);
		}


		v.setZero(); v.template loadWithOffset<3, 2, HK_IO_NATIVE_ALIGNED>(base);
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(cc->val[i+0] == s_mxvector_aligned_data[i+0]); 
			HK_TEST(cc->val[i+1] == s_mxvector_aligned_data[i+1]); 
			//HK_TEST(cc[i+2] == 0.0f); HK_TEST(cc[i+3] == 0.0f);
		}

		v.setZero(); v.template loadWithOffset<3, 3, HK_IO_NATIVE_ALIGNED>(base);
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(cc->val[i+0] == s_mxvector_aligned_data[i+0]); 
			HK_TEST(cc->val[i+1] == s_mxvector_aligned_data[i+1]); 
			HK_TEST(cc->val[i+2] == s_mxvector_aligned_data[i+2]); 
			//HK_TEST(cc[i+3] == 0.0f);
		}

		v.setZero(); v.template loadWithOffset<3, 4, HK_IO_NATIVE_ALIGNED>(base);
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(cc->val[i+0] == s_mxvector_aligned_data[i+0]); 
			HK_TEST(cc->val[i+1] == s_mxvector_aligned_data[i+1]); 
			HK_TEST(cc->val[i+2] == s_mxvector_aligned_data[i+2]); 
			HK_TEST(cc->val[i+3] == s_mxvector_aligned_data[i+3]); 
		}
	}
	{
		// Not loaded components are undefined.

		hkHalf half_data[32]; 
		for (int i=0; i<32; ++i)
		{
			half_data[i].setReal<false>(s_mxvector_aligned_data[i]);
		}

		hkMxVector<M> v; 
		const _MXVector<M>* c = (const _MXVector<M>*)&(v.m_vec);

		v.setZero(); v.template loadUnpack<4*sizeof(hkHalf), 1, HK_IO_NATIVE_ALIGNED>(half_data);
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c->val[i+0] == s_mxvector_aligned_data[i+0]); 
			//HK_TEST(c[i+1] == 0.0f); HK_TEST(c[i+2] == 0.0f); HK_TEST(c[i+3] == 0.0f);
		}

		v.setZero(); v.template loadUnpack<4*sizeof(hkHalf), 2, HK_IO_NATIVE_ALIGNED>(half_data);
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c->val[i+0] == s_mxvector_aligned_data[i+0]); 
			HK_TEST(c->val[i+1] == s_mxvector_aligned_data[i+1]); 
			//HK_TEST(c[i+2] == 0.0f); HK_TEST(c[i+3] == 0.0f);
		}

		v.setZero(); v.template loadUnpack<4*sizeof(hkHalf), 3, HK_IO_NATIVE_ALIGNED>(half_data);
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c->val[i+0] == s_mxvector_aligned_data[i+0]); 
			HK_TEST(c->val[i+1] == s_mxvector_aligned_data[i+1]); 
			HK_TEST(c->val[i+2] == s_mxvector_aligned_data[i+2]); 
			//HK_TEST(c[i+3] == 0.0f);
		}

		v.setZero(); v.template loadUnpack<4*sizeof(hkHalf), 4, HK_IO_NATIVE_ALIGNED>(half_data);
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c->val[i+0] == s_mxvector_aligned_data[i+0]); 
			HK_TEST(c->val[i+1] == s_mxvector_aligned_data[i+1]); 
			HK_TEST(c->val[i+2] == s_mxvector_aligned_data[i+2]); 
			HK_TEST(c->val[i+3] == s_mxvector_aligned_data[i+3]); 
		}
	}
	{
		// Not loaded components are undefined.

		hkHalf half_data[32]; 
		for (int i=0; i<32; ++i)
		{
			half_data[i].setReal<false>(s_mxvector_aligned_data[i]);
		}

		const hkHalf* base[M]; for (int i=0; i<M; i++) base[i] = (hkHalf*) (((char*)(half_data+i*4))-3);
		hkMxVector<M> v; 
		const _MXVector<M>* cc = (const _MXVector<M>*)&(v.m_vec);

		v.setZero(); v.template loadUnpackWithOffset<3, 1, HK_IO_NATIVE_ALIGNED>(base);
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(cc->val[i+0] == s_mxvector_aligned_data[i+0]); 
			//HK_TEST(cc[i+1] == 0.0f); HK_TEST(cc[i+2] == 0.0f); HK_TEST(cc[i+3] == 0.0f);
		}


		v.setZero(); v.template loadUnpackWithOffset<3, 2, HK_IO_NATIVE_ALIGNED>(base);
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(cc->val[i+0] == s_mxvector_aligned_data[i+0]); 
			HK_TEST(cc->val[i+1] == s_mxvector_aligned_data[i+1]); 
			//HK_TEST(cc[i+2] == 0.0f); HK_TEST(cc[i+3] == 0.0f);
		}

		v.setZero(); v.template loadUnpackWithOffset<3, 3, HK_IO_NATIVE_ALIGNED>(base);
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(cc->val[i+0] == s_mxvector_aligned_data[i+0]); 
			HK_TEST(cc->val[i+1] == s_mxvector_aligned_data[i+1]); 
			HK_TEST(cc->val[i+2] == s_mxvector_aligned_data[i+2]); 
			//HK_TEST(cc[i+3] == 0.0f);
		}

		v.setZero(); v.template loadUnpackWithOffset<3, 4, HK_IO_NATIVE_ALIGNED>(base);
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(cc->val[i+0] == s_mxvector_aligned_data[i+0]); 
			HK_TEST(cc->val[i+1] == s_mxvector_aligned_data[i+1]); 
			HK_TEST(cc->val[i+2] == s_mxvector_aligned_data[i+2]); 
			HK_TEST(cc->val[i+3] == s_mxvector_aligned_data[i+3]); 
		}
	}
#endif
}

template <int M>
static void mxvector_store_assign()
{
	HK_ALIGN_REAL(hkUint8 mem[512]);

	hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);

	{
		const hkUint8* dataStart = mem+2*sizeof(hkVector4);
		hkMemUtil::memSet(mem, 0x99, 512);
		v.moveStore((hkVector4*)dataStart);
		for (int i=0; i<2*hkSizeOf(hkVector4); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		const hkReal* c = (const hkReal*)dataStart;
		for (int i=0; i<M*4; ++i)
		{
			HK_TEST(c[i] == s_mxvector_aligned_data[i]);
		}
		for (int i=sizeof(hkVector4)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
	}	
	{
		const hkUint8* dataStart = mem+2*sizeof(hkVector4);
		hkMemUtil::memSet(mem, 0x99, 512);
		v.store((hkReal*)dataStart);
		for (int i=0; i<2*hkSizeOf(hkVector4); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		const hkReal* c = (const hkReal*)dataStart;
		for (int i=0; i<M*4; ++i)
		{
			HK_TEST(c[i] == s_mxvector_aligned_data[i]);
		}
		for (int i=sizeof(hkVector4)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
	}	
	{
		const hkUint8* dataStart = mem+2*sizeof(hkVector4);
		hkMemUtil::memSet(mem, 0x99, 512);
		v.storeNotCached((hkReal*)dataStart);
		for (int i=0; i<2*hkSizeOf(hkVector4); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		const hkReal* c = (const hkReal*)dataStart;
		for (int i=0; i<M*4; ++i)
		{
			HK_TEST(c[i] == s_mxvector_aligned_data[i]);
		}
		for (int i=sizeof(hkVector4)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
	}	
#if 0
	// byte-aligned not supported on all platforms yet
	!defined(HK_PLATFORM_PS3)
	{
		const hkUint8* dataStart = mem+3;
		hkMemUtil::memSet(mem, 0x99, 512);
		v.storeNotAligned((hkReal*)dataStart);
		for (int i=0; i<3; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		const hkReal* c = (const hkReal*)dataStart;
		for (int i=0; i<M*4; ++i)
		{
			HK_TEST(c[i] == s_mxvector_aligned_data[i]);
		}
		for (int i=3+(sizeof(hkVector4)*M); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
	}	
#endif

#if (HK_ENDIAN_LITTLE == 1) && !defined(MX_VECTOR_IS_AVX) && !defined(HK_REAL_IS_DOUBLE)
	// The following tests use bit manipulation and byte comparison, so they are only enabled for a limited number of configurations.

	{
		int check_hex = (0x99 | (0x99 << 8) | (0x99 << 16) | (0x99 << 24));
		hkReal check_real = *((hkReal*)(&check_hex));
		const hkUint8* dataStart = mem+2*4*sizeof(hkReal);
		hkReal* c = (hkReal*)dataStart;

		hkMemUtil::memSet(mem, 0x99, 512);
		v.template store<4*hkSizeOf(hkReal), 1, HK_IO_NATIVE_ALIGNED>(c);
		for (int i=0; i<2*4*hkSizeOf(hkReal); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c[i+1] == check_real ); HK_TEST(c[i+2] == check_real ); HK_TEST(c[i+3] == check_real );
		}
		for (int i=4*sizeof(hkReal)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}

		hkMemUtil::memSet(mem, 0x99, 512);
		v.template store<4*hkSizeOf(hkReal), 2, HK_IO_NATIVE_ALIGNED>(c);
		for (int i=0; i<2*4*hkSizeOf(hkReal); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c[i+1] == s_mxvector_aligned_data[i+1]);
			HK_TEST(c[i+2] == check_real ); HK_TEST(c[i+3] == check_real );
		}
		for (int i=4*sizeof(hkReal)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}

		hkMemUtil::memSet(mem, 0x99, 512);
		v.template store<4*hkSizeOf(hkReal), 3, HK_IO_NATIVE_ALIGNED>(c);
		for (int i=0; i<2*4*hkSizeOf(hkReal); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c[i+1] == s_mxvector_aligned_data[i+1]);
			HK_TEST(c[i+2] == s_mxvector_aligned_data[i+2]);
			HK_TEST(c[i+3] == check_real );
		}
		for (int i=4*sizeof(hkReal)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}

		hkMemUtil::memSet(mem, 0x99, 512);
		v.template store<4*hkSizeOf(hkReal), 4, HK_IO_NATIVE_ALIGNED>(c);
		for (int i=0; i<2*4*hkSizeOf(hkReal); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c[i+1] == s_mxvector_aligned_data[i+1]);
			HK_TEST(c[i+2] == s_mxvector_aligned_data[i+2]);
			HK_TEST(c[i+3] == s_mxvector_aligned_data[i+3]);
		}
		for (int i=4*hkSizeOf(hkReal)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
	}
	{
		int check_hex = (0x99 | (0x99 << 8) | (0x99 << 16) | (0x99 << 24));
		hkReal check_real = *((hkReal*)(&check_hex));
		const hkUint8* dataStart = mem+2*4*hkSizeOf(hkReal);
		hkReal* c = (hkReal*)dataStart;

		hkReal* base[M]; for (int i=0; i<M; i++) base[i] = (hkReal*) (((char*)(c+i*4))-3);

		hkMemUtil::memSet(mem, 0x99, 512);
		v.template storeWithOffset<3, 1, HK_IO_NATIVE_ALIGNED>(base);
		for (int i=0; i<2*4*hkSizeOf(hkReal); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c[i+1] == check_real ); HK_TEST(c[i+2] == check_real ); HK_TEST(c[i+3] == check_real );
		}
		for (int i=4*hkSizeOf(hkReal)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}

		hkMemUtil::memSet(mem, 0x99, 512);
		v.template storeWithOffset<3, 2, HK_IO_NATIVE_ALIGNED>(base);
		for (int i=0; i<2*4*hkSizeOf(hkReal); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c[i+1] == s_mxvector_aligned_data[i+1]);
			HK_TEST(c[i+2] == check_real ); HK_TEST(c[i+3] == check_real );
		}
		for (int i=4*hkSizeOf(hkReal)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}

		hkMemUtil::memSet(mem, 0x99, 512);
		v.template storeWithOffset<3, 3, HK_IO_NATIVE_ALIGNED>(base);
		for (int i=0; i<2*4*hkSizeOf(hkReal); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c[i+1] == s_mxvector_aligned_data[i+1]);
			HK_TEST(c[i+2] == s_mxvector_aligned_data[i+2]);
			HK_TEST(c[i+3] == check_real );
		}
		for (int i=4*hkSizeOf(hkReal)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}

		hkMemUtil::memSet(mem, 0x99, 512);
		v.template storeWithOffset<3, 4, HK_IO_NATIVE_ALIGNED>(base);
		for (int i=0; i<2*4*hkSizeOf(hkReal); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c[i+1] == s_mxvector_aligned_data[i+1]);
			HK_TEST(c[i+2] == s_mxvector_aligned_data[i+2]);
			HK_TEST(c[i+3] == s_mxvector_aligned_data[i+3]);
		}
		for (int i=4*hkSizeOf(hkReal)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
	}

	{
		int check_hex = (0x99 | (0x99 << 8));
		hkHalf check_half = *((hkHalf*)(&check_hex));
		const hkUint8* dataStart = mem+2*4*hkSizeOf(hkHalf);
		hkHalf* c = (hkHalf*)dataStart;

		hkMemUtil::memSet(mem, 0x99, 512);
		v.template storePack<4*hkSizeOf(hkHalf), 1, HK_IO_NATIVE_ALIGNED, HK_ROUND_NEAREST>(c);
		for (int i=0; i<2*4*hkSizeOf(hkHalf); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c[i+1] == check_half ); HK_TEST(c[i+2] == check_half ); HK_TEST(c[i+3] == check_half );
		}
		for (int i=4*hkSizeOf(hkHalf)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}

		hkMemUtil::memSet(mem, 0x99, 512);
		v.template storePack<4*hkSizeOf(hkHalf), 2, HK_IO_NATIVE_ALIGNED, HK_ROUND_NEAREST>(c);
		for (int i=0; i<2*4*hkSizeOf(hkHalf); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c[i+1] == s_mxvector_aligned_data[i+1]);
			HK_TEST(c[i+2] == check_half ); HK_TEST(c[i+3] == check_half );
		}
		for (int i=4*hkSizeOf(hkHalf)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}

		hkMemUtil::memSet(mem, 0x99, 512);
		v.template storePack<4*hkSizeOf(hkHalf), 3, HK_IO_NATIVE_ALIGNED, HK_ROUND_NEAREST>(c);
		for (int i=0; i<2*4*hkSizeOf(hkHalf); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c[i+1] == s_mxvector_aligned_data[i+1]);
			HK_TEST(c[i+2] == s_mxvector_aligned_data[i+2]);
			HK_TEST(c[i+3] == check_half );
		}
		for (int i=4*hkSizeOf(hkHalf)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}

		hkMemUtil::memSet(mem, 0x99, 512);
		v.template storePack<4*hkSizeOf(hkHalf), 4, HK_IO_NATIVE_ALIGNED, HK_ROUND_NEAREST>(c);
		for (int i=0; i<2*4*hkSizeOf(hkHalf); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c[i+1] == s_mxvector_aligned_data[i+1]);
			HK_TEST(c[i+2] == s_mxvector_aligned_data[i+2]);
			HK_TEST(c[i+3] == s_mxvector_aligned_data[i+3]);
		}
		for (int i=4*hkSizeOf(hkHalf)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
	}
	{
		int check_hex = (0x99 | (0x99 << 8));
		hkHalf check_half = *((hkHalf*)(&check_hex));
		const hkUint8* dataStart = mem+2*4*hkSizeOf(hkHalf);
		hkHalf* c = (hkHalf*)dataStart;

		hkHalf* base[M]; for (int i=0; i<M; i++) base[i] = (hkHalf*) (((char*)(c+i*4))-3);

		hkMemUtil::memSet(mem, 0x99, 512);
		v.template storePackWithOffset<3, 1, HK_IO_NATIVE_ALIGNED, HK_ROUND_NEAREST>(base);
		for (int i=0; i<2*4*hkSizeOf(hkHalf); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c[i+1] == check_half ); HK_TEST(c[i+2] == check_half ); HK_TEST(c[i+3] == check_half );
		}
		for (int i=4*hkSizeOf(hkHalf)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}

		hkMemUtil::memSet(mem, 0x99, 512);
		v.template storePackWithOffset<3, 2, HK_IO_NATIVE_ALIGNED, HK_ROUND_NEAREST>(base);
		for (int i=0; i<2*4*hkSizeOf(hkHalf); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c[i+1] == s_mxvector_aligned_data[i+1]);
			HK_TEST(c[i+2] == check_half ); HK_TEST(c[i+3] == check_half );
		}
		for (int i=4*hkSizeOf(hkHalf)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}

		hkMemUtil::memSet(mem, 0x99, 512);
		v.template storePackWithOffset<3, 3, HK_IO_NATIVE_ALIGNED, HK_ROUND_NEAREST>(base);
		for (int i=0; i<2*4*hkSizeOf(hkHalf); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c[i+1] == s_mxvector_aligned_data[i+1]);
			HK_TEST(c[i+2] == s_mxvector_aligned_data[i+2]);
			HK_TEST(c[i+3] == check_half );
		}
		for (int i=4*hkSizeOf(hkHalf)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}

		hkMemUtil::memSet(mem, 0x99, 512);
		v.template storePackWithOffset<3, 4, HK_IO_NATIVE_ALIGNED, HK_ROUND_NEAREST>(base);
		for (int i=0; i<2*4*hkSizeOf(hkHalf); ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
		for (int i=0; i<M*4; i+=4)
		{
			HK_TEST(c[i+0] == s_mxvector_aligned_data[i+0]);
			HK_TEST(c[i+1] == s_mxvector_aligned_data[i+1]);
			HK_TEST(c[i+2] == s_mxvector_aligned_data[i+2]);
			HK_TEST(c[i+3] == s_mxvector_aligned_data[i+3]);
		}
		for (int i=4*hkSizeOf(hkHalf)*(M+2); i<512; ++i)
		{
			HK_TEST(mem[i] == 0x99);
		}
	}
#endif // defined(HK_ENDIAN_LITTLE) !defined(MX_VECTOR_IS_AVX) && !defined(HK_REAL_IS_DOUBLE)
}

template <int M>
static void mxvector_setPermutation()
{
	hkMxVector<M> v; v.template load((const hkReal*)s_mxvector_aligned_data);

	{
		hkMxVector<M> p; p.template setVectorPermutation<hkMxVectorPermutation::REVERSE>(v);
		const hkVector4* c = (const hkVector4*)&(p.m_vec);
		const hkVector4* dataStart = (const hkVector4*)s_mxvector_aligned_data;
		for (int i=0; i<M; ++i)
		{
			HK_TEST(c[i](0) == dataStart[M-1-i](0));
			HK_TEST(c[i](1) == dataStart[M-1-i](1));
			HK_TEST(c[i](2) == dataStart[M-1-i](2));
			HK_TEST(c[i](3) == dataStart[M-1-i](3));
		}
	}
	{
		hkMxVector<M> p; p.template setVectorPermutation<hkMxVectorPermutation::SHIFT_LEFT_CYCLIC>(v);
		const hkVector4* c = (const hkVector4*)&(p.m_vec);
		const hkVector4* dataStart = (const hkVector4*)s_mxvector_aligned_data;
		for (int i=0; i<M; ++i)
		{
			HK_TEST(c[i](0) == dataStart[(i+1)%M](0));
			HK_TEST(c[i](1) == dataStart[(i+1)%M](1));
			HK_TEST(c[i](2) == dataStart[(i+1)%M](2));
			HK_TEST(c[i](3) == dataStart[(i+1)%M](3));
		}
	}
	{
		hkMxVector<M> p; p.template setVectorPermutation<hkMxVectorPermutation::SHIFT_RIGHT_CYCLIC>(v);
		const hkVector4* c = (const hkVector4*)&(p.m_vec);
		const hkVector4* dataStart = (const hkVector4*)s_mxvector_aligned_data;
		for (int i=0; i<M; ++i)
		{
			HK_TEST(c[(i+1)%M](0) == dataStart[i](0));
			HK_TEST(c[(i+1)%M](1) == dataStart[i](1));
			HK_TEST(c[(i+1)%M](2) == dataStart[i](2));
			HK_TEST(c[(i+1)%M](3) == dataStart[i](3));
		}
	}
}

template <int M, int N>
static void mxvector_horizontal()
{
	hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);

	{
		hkVector4 ad; v.template horizontalAdd<N>(ad);
		int c = 0; for (int i=0;i<N;++i) c+=i;
		for (int i=0; i<M; ++i)
		{
			HK_TEST(ad(i) == hkReal(N*i*4+c));
		}
	}
	{
		hkVector4 ad; v.template horizontalMin<N>(ad);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(ad(i) == hkReal(i*4));
		}
	}
	{
		hkVector4 ad; v.template horizontalMax<N>(ad);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(ad(i) == hkReal(i*4+N-1));
		}
	}
}

template <int M>
static void mxvector_setAsBC()
{
	hkVector4 bc; bc.set(10,20,30,40);
	hkMxVector<M> v; v.setAsBroadcast(bc);
	const _MXVector<M>* c = (const _MXVector<M>*)&(v.m_vec);
	for (int i=0; i<M; ++i)
	{
		for (int j=0; j<4; j++)
		{
			HK_TEST(c->val[(i*4)+j] == bc(i));
		}
	}
}

template <int M>
static void mxvector_setAddMul()
{	
	hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);
	hkMxVector<M> w; w.template load<4>((const hkReal*)(s_mxvector_aligned_data+16));
	hkVector4 s; s.setAll(2);

	hkMxVector<M> x; x.setAddMul(v,w,s);

	const _MXVector<M>* c = (const _MXVector<M>*)&(x.m_vec);
	for (int i=0; i<M; ++i)
	{
		for (int j=0; j<4; j++)
		{
			HK_TEST(c->val[(i*4)+j] == (i*4)+j+(((i*4)+j)*2));
		}
	}
}

template <int M>
static void mxvector_storeTransposed4()
{
	hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);
	hkMatrix4 m; v.storeTransposed4(m);
	hkMatrix4 t; t.setTranspose(m);

	const _MXVector<M>* cv = (const _MXVector<M>*)&(v.m_vec);
	const _MXVector<M>* cm = (const _MXVector<M>*)&t;
	for (int i=0; i<M; ++i)
	{
		for (int j=0; j<4; j++)
		{
			HK_TEST(cv->val[(i*4)+j] == cm->val[(i*4)+j]);
		}
	}
}

template <int M, int N>
static void mxvector_isOk()
{
	hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);
	{
		hkBool32 ok = v.template isOk<N>();
		HK_TEST(ok);
	}
}

template <int M>
static void mxvector_reduceAdd()
{
	hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);
	hkVector4 a; v.reduceAdd(a);
	hkVector4 r; r.setZero();
	for (int i=0; i<M; ++i)
	{
		hkVector4 s; s.set(hkReal(i*4),hkReal(i*4+1),hkReal(i*4+2),hkReal(i*4+3));
		r.add(s);
	}
	HK_TEST(a(0) == r(0));
	HK_TEST(a(1) == r(1));
	HK_TEST(a(2) == r(2));
	HK_TEST(a(3) == r(3));
}

template <int M, int N>
static void mxvector_dot()
{
	hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);
	hkMxVector<M> w; w.template load<4>((const hkReal*)(s_mxvector_aligned_data+16));

	{
		hkMxReal<M> d; v.template dot<N>(w,d);
		const hkSimdReal* dd = (const hkSimdReal*)&(d.m_real);
		for (int i=0; i<M; ++i)
		{
			hkVector4 a; a.set(hkReal(i*4),hkReal(i*4+1),hkReal(i*4+2),hkReal(i*4+3));
			const hkSimdReal t = a.dot<N>(a);
			HK_TEST(t.getReal() == dd[i].getReal());
		}
	}
}

template <int M, int N>
static void mxvector_length()
{
	hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);

	{
		hkMxReal<M> l; v.template length<N,HK_ACC_FULL,HK_SQRT_SET_ZERO>(l);
		const hkSimdReal* ll = (const hkSimdReal*)&(l.m_real);
		for (int i=0; i<M; ++i)
		{
			hkVector4 a; a.set(hkReal(i*4),hkReal(i*4+1),hkReal(i*4+2),hkReal(i*4+3));
			const hkSimdReal t = a.length<N,HK_ACC_FULL,HK_SQRT_SET_ZERO>();
			HK_TEST(t.getReal() == ll[i].getReal());
		}
	}
	{
		hkMxReal<M> l; v.template length<N,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>(l);
		const hkSimdReal* ll = (const hkSimdReal*)&(l.m_real);
		for (int i=0; i<M; ++i)
		{
			hkVector4 a; a.set(hkReal(i*4),hkReal(i*4+1),hkReal(i*4+2),hkReal(i*4+3));
			const hkSimdReal t = a.length<N,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>();
			HK_TEST(t.getReal() == ll[i].getReal());
		}
	}
	{
		hkMxReal<M> l; v.template length<N,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>(l);
		const hkSimdReal* ll = (const hkSimdReal*)&(l.m_real);
		for (int i=0; i<M; ++i)
		{
			hkVector4 a; a.set(hkReal(i*4),hkReal(i*4+1),hkReal(i*4+2),hkReal(i*4+3));
			const hkSimdReal t = a.length<N,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>();
			HK_TEST(t.getReal() == ll[i].getReal());
		}
	}
	{
		hkMxReal<M> l; v.template lengthSquared<N>(l);
		const hkSimdReal* ll = (const hkSimdReal*)&(l.m_real);
		for (int i=0; i<M; ++i)
		{
			hkVector4 a; a.set(hkReal(i*4),hkReal(i*4+1),hkReal(i*4+2),hkReal(i*4+3));
			const hkSimdReal t = a.lengthSquared<N>();
			HK_TEST(t.getReal() == ll[i].getReal());
		}
	}
	{
		hkMxReal<M> l; v.template lengthInverse<N,HK_ACC_FULL,HK_SQRT_SET_ZERO>(l);
		const hkSimdReal* ll = (const hkSimdReal*)&(l.m_real);
		for (int i=0; i<M; ++i)
		{
			hkVector4 a; a.set(hkReal(i*4),hkReal(i*4+1),hkReal(i*4+2),hkReal(i*4+3));
			const hkSimdReal t = a.lengthInverse<N,HK_ACC_FULL,HK_SQRT_SET_ZERO>();
			HK_TEST(t.getReal() == ll[i].getReal());
		}
	}
	{
		hkMxReal<M> l; v.template lengthInverse<N,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>(l);
		const hkSimdReal* ll = (const hkSimdReal*)&(l.m_real);
		for (int i=0; i<M; ++i)
		{
			hkVector4 a; a.set(hkReal(i*4),hkReal(i*4+1),hkReal(i*4+2),hkReal(i*4+3));
			const hkSimdReal t = a.lengthInverse<N,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>();
			HK_TEST(t.getReal() == ll[i].getReal());
		}
	}
	{
		hkMxReal<M> l; v.template lengthInverse<N,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>(l);
		const hkSimdReal* ll = (const hkSimdReal*)&(l.m_real);
		for (int i=0; i<M; ++i)
		{
			hkVector4 a; a.set(hkReal(i*4),hkReal(i*4+1),hkReal(i*4+2),hkReal(i*4+3));
			const hkSimdReal t = a.lengthInverse<N,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>();
			HK_TEST(t.getReal() == ll[i].getReal());
		}
	}
}

template <int M, int C>
static void mxvector_constant()
{
	hkMxVector<M> v; v.template setConstant<C>();
	const _MXVector<M>* vv = (const _MXVector<M>*)&(v.m_vec);
	hkVector4 c = hkVector4::getConstant<C>();
	for (int i=0; i<M; ++i)
	{
		for (int j=0; j<4; ++j)
		{
			HK_TEST(vv->val[i*4+j] == c(j));
		}
	}
}

template <int M>
struct SetMask
{
	hkMxMask<M>* mask;

	template <int I>
	void step()
	{
		hkVector4Comparison m;
		if (I%2)
		{
			m.set<hkVector4ComparisonMask::MASK_XYZW>();
		}
		else
		{
			m.set<hkVector4ComparisonMask::MASK_NONE>();
		}
		mask->template set<I>(m);
	}
};

template <int M>
static void mxvector_select()
{
	hkMxVector<M> one; one.template setConstant<HK_QUADREAL_1>();
	hkMxVector<M> two; two.template setConstant<HK_QUADREAL_2>();

	{
		hkMxMask<M> mask;
		mask.template setAll<hkVector4ComparisonMask::MASK_XYZW>();
		hkMxVector<M> v; v.setSelect(mask,one,two);
		const _MXVector<M>* vv = (const _MXVector<M>*)&(v.m_vec);
		for (int i=0; i<4*M; ++i)
		{
			HK_TEST(vv->val[i] == 1.0f);
		}
	}
	{
		hkMxMask<M> mask;
		SetMask<M> setMask;
		setMask.mask = &mask;
		hkMxUnroller<0,M>::step(setMask);
		hkMxVector<M> v; v.setSelect(mask,one,two);
		const _MXVector<M>* vv = (const _MXVector<M>*)&(v.m_vec);
		for (int m=0; m<M; ++m)
		{
			for (int i=0; i<4; ++i)
			{
				HK_TEST(vv->val[m*4+i] == ((m%2) ? 1.0f : 2.0f));
			}
		}
	}
}

template <int M>
static void mxvector_compare()
{
	hkMxVector<M> zero; zero.template setConstant<HK_QUADREAL_0>();
	hkMxVector<M> minusZero;
#if defined(HK_REAL_IS_DOUBLE)
	{
		hkInt64* vmz = (hkInt64*)&(minusZero.m_vec);
		for (int i=0; i<4*M; ++i) vmz[i] = 0x8000000000000000ull;
	}
#else
	{
		hkUint32* vmz = (hkUint32*)&(minusZero.m_vec);
		for (int i=0; i<4*M; ++i) vmz[i] = 0x80000000;
	}
#endif
	hkMxVector<M> one; one.template setConstant<HK_QUADREAL_1>();
	hkMxVector<M> two; two.template setConstant<HK_QUADREAL_2>();

	{
		hkMxMask<M> mask;
		zero.equal(minusZero, mask);
		HK_TEST( mask.template get<0>().allAreSet() );
		zero.equal(one, mask);
		HK_TEST( ! mask.anyIsSet() );
	}
	{
		hkMxMask<M> mask;
		two.notEqual(minusZero, mask);
		HK_TEST( mask.template get<0>().allAreSet() );
		zero.notEqual(minusZero, mask);
		HK_TEST( ! mask.anyIsSet() );
	}
	{
		hkMxMask<M> mask;
		zero.less(one, mask);
		HK_TEST( mask.template get<0>().allAreSet() );
		one.less(one, mask);
		HK_TEST( ! mask.anyIsSet() );
	}
	{
		hkMxMask<M> mask;
		zero.lessEqual(minusZero, mask);
		HK_TEST( mask.template get<0>().allAreSet() );
		two.lessEqual(one, mask);
		HK_TEST( ! mask.anyIsSet() );
	}
	{
		hkMxMask<M> mask;
		zero.greaterEqual(minusZero, mask);
		HK_TEST( mask.template get<0>().allAreSet() );
		zero.greaterEqual(one, mask);
		HK_TEST( ! mask.anyIsSet() );
	}
	{
		hkMxMask<M> mask;
		two.greater(one, mask);
		HK_TEST( mask.template get<0>().allAreSet() );
		one.greater(one, mask);
		HK_TEST( ! mask.anyIsSet() );
	}
}

template <int M>
static void mxvector_setInterpolate()
{
	hkMxVector<M> one; one.template setConstant<HK_QUADREAL_1>();
	hkMxVector<M> two; two.template setConstant<HK_QUADREAL_2>();
	hkMxReal<M> half; half.setBroadcast(hkSimdReal::getConstant<HK_QUADREAL_INV_2>());

	hkMxVector<M> v; v.setInterpolate(one,two,half);
	const _MXVector<M>* vv = (const _MXVector<M>*)&(v.m_vec);
	for (int i=0; i<4*M; ++i)
	{
		HK_TEST( (vv->val[i] > 1.4999f) && (vv->val[i] < 1.5001f) );
	}
}

template <int M>
static void mxvector_minMaxNegAbs()
{
	hkMxVector<M> one; one.template setConstant<HK_QUADREAL_1>();
	hkMxVector<M> two; two.template setConstant<HK_QUADREAL_2>();

	hkMxVector<M> v; v.setMin(one,two);
	const _MXVector<M>* vv = (const _MXVector<M>*)&(v.m_vec);
	for (int i=0; i<4*M; ++i)
	{
		HK_TEST( vv->val[i] == 1.0f );
	}
	v.setMax(one,two);
	for (int i=0; i<4*M; ++i)
	{
		HK_TEST( vv->val[i] == 2.0f );
	}
	v.template setNeg<4>(one);
	for (int i=0; i<4*M; ++i)
	{
		HK_TEST( vv->val[i] == -1.0f );
	}
	v.template setNeg<2>(two);
	for (int i=0; i<M; ++i)
	{
		HK_TEST( vv->val[i*4] == -2.0f );
		HK_TEST( vv->val[i*4+1] == -2.0f );
		HK_TEST( vv->val[i*4+2] ==  2.0f );
		HK_TEST( vv->val[i*4+3] ==  2.0f );
	}
	hkMxVector<M> w; w.setAbs(v);
	const _MXVector<M>* ww = (const _MXVector<M>*)&(w.m_vec);
	for (int i=0; i<4*M; ++i)
	{
		HK_TEST( ww->val[i] == 2.0f );
	}
}

template <int M>
static void mxvector_cross()
{
	hkMxVector<M> x; x.template load<4>((const hkReal*)s_mxvector_aligned_data);
	hkMxVector<M> y; y.template load<4>((const hkReal*)(s_mxvector_aligned_data+16));

	hkMxVector<M> v; v.setCross(x,y);
	const _MXVector<M>* vv = (const _MXVector<M>*)&(v.m_vec);

	for (int i=0; i<M; ++i)
	{
		hkVector4 xx; xx.set(hkReal(i*4),hkReal(i*4+1),hkReal(i*4+2),hkReal(i*4+3));
		hkVector4 c; c.setCross(xx,xx);

		HK_TEST( vv->val[i*4  ] == c(0) );
		HK_TEST( vv->val[i*4+1] == c(1) );
		HK_TEST( vv->val[i*4+2] == c(2) );
	}
}

template <int M>
static void mxvector_calc()
{
	hkMxVector<M> x; x.template load<4>((const hkReal*)s_mxvector_aligned_data);
	hkMxVector<M> two; two.template setConstant<HK_QUADREAL_2>();

	hkMxVector<M> v; 
	const _MXVector<M>* vv = (const _MXVector<M>*)&(v.m_vec);
	{
		v.setAdd(x,two);
		for (int i=0; i<4*M; ++i)
		{
			HK_TEST( vv->val[i] == (s_mxvector_aligned_data[i] + 2.0f) );
		}
	}
	{
		v.setSub(x,two);
		for (int i=0; i<4*M; ++i)
		{
			HK_TEST( vv->val[i] == (s_mxvector_aligned_data[i] - 2.0f) );
		}
	}
	{
		v.setMul(x,two);
		for (int i=0; i<4*M; ++i)
		{
			HK_TEST( vv->val[i] == (s_mxvector_aligned_data[i] * 2.0f) );
		}
	}
	{
		v.setAddMul(x,two,two);
		for (int i=0; i<4*M; ++i)
		{
			HK_TEST( vv->val[i] == (s_mxvector_aligned_data[i] + 4.0f) );
		}
	}
	{
		v.setSubMul(x,two,two);
		for (int i=0; i<4*M; ++i)
		{
			HK_TEST( vv->val[i] == (s_mxvector_aligned_data[i] - 4.0f) );
		}
	}
	{
		x.template zeroComponent<2>();
		const _MXVector<M>* xx = (const _MXVector<M>*)&(x.m_vec);
		for (int i=0; i<M; ++i)
		{
			HK_TEST( xx->val[i*4] == hkReal(i*4) );
			HK_TEST( xx->val[i*4+1] == hkReal(i*4+1) );
			HK_TEST( xx->val[i*4+2] == 0.0f );
			HK_TEST( xx->val[i*4+3] == hkReal(i*4+3) );
		}
	}
	{
		x.setZero();
		const _MXVector<M>* xx = (const _MXVector<M>*)&(x.m_vec);
		for (int i=0; i<4*M; ++i)
		{
			HK_TEST( xx->val[i] == 0.0f );
		}
	}
}

template <int M>
static void mxvector_xyzw()
{
	hkMxVector<M> x; x.template load<4>((const hkReal*)s_mxvector_aligned_data);
	hkMxVector<M> c; c.template setConstant<HK_QUADREAL_MINUS1>();

	hkMxVector<M> v; v.setXYZ_W(x,c);
	const _MXVector<M>* vv = (const _MXVector<M>*)&(v.m_vec);

	for (int i=0; i<M; ++i)
	{
		HK_TEST( vv->val[i*4  ] == hkReal(i*4) );
		HK_TEST( vv->val[i*4+1] == hkReal(i*4+1) );
		HK_TEST( vv->val[i*4+2] == hkReal(i*4+2) );
		HK_TEST( vv->val[i*4+3] == -1.0f );
	}
}

template <int M, int N>
static void mxvector_normalize()
{
	{
		hkMxVector<M> x; x.template load<4>((const hkReal*)s_mxvector_aligned_data);
		x.template normalize<N,HK_ACC_FULL,HK_SQRT_SET_ZERO>();
		const _MXVector<M>* xx = (const _MXVector<M>*)&(x.m_vec);
		for (int i=0; i<M; ++i)
		{
			hkVector4 y; y.set(hkReal(i*4),hkReal(i*4+1),hkReal(i*4+2),hkReal(i*4+3)); y.normalize<N,HK_ACC_FULL,HK_SQRT_SET_ZERO>();
			for (int j=0; j<N; ++j)
			{
				HK_TEST( xx->val[i*4+j] == y(j) );
			}
		}
	}
	{
		hkMxVector<M> x; x.template load<4>((const hkReal*)s_mxvector_aligned_data);
		x.template normalize<N,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>();
		const _MXVector<M>* xx = (const _MXVector<M>*)&(x.m_vec);
		for (int i=0; i<M; ++i)
		{
			hkVector4 y; y.set(hkReal(i*4),hkReal(i*4+1),hkReal(i*4+2),hkReal(i*4+3)); y.normalize<N,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>();
			for (int j=0; j<N; ++j)
			{
				HK_TEST( xx->val[i*4+j] == y(j) );
			}
		}
	}
	{
		hkMxVector<M> x; x.template load<4>((const hkReal*)s_mxvector_aligned_data);
		x.template normalize<N,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>();
		const _MXVector<M>* xx = (const _MXVector<M>*)&(x.m_vec);
		for (int i=0; i<M; ++i)
		{
			hkVector4 y; y.set(hkReal(i*4),hkReal(i*4+1),hkReal(i*4+2),hkReal(i*4+3)); y.normalize<N,HK_ACC_23_BIT,HK_SQRT_SET_ZERO>();
			for (int j=0; j<N; ++j)
			{
				HK_TEST( xx->val[i*4+j] == y(j) );
			}
		}
	}
}

template <int M>
static void mxvector_sqrt()
{
	hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);
	hkMxVector<M> one; one.template setConstant<HK_QUADREAL_1>();
	v.add(one);
	{
		hkMxVector<M> x; x.template setSqrtInverse<HK_ACC_FULL,HK_SQRT_SET_ZERO>(v);
		const _MXVector<M>* xx = (const _MXVector<M>*)&(x.m_vec);
		for (int i=0; i<M; ++i)
		{
			hkVector4 w; w.set(hkReal(i*4+1),hkReal(i*4+2),hkReal(i*4+3),hkReal(i*4+4)); 
			hkVector4 y; y.setSqrtInverse<HK_ACC_FULL,HK_SQRT_SET_ZERO>(w);
			HK_TEST( xx->val[i*4  ] == y(0) );
			HK_TEST( xx->val[i*4+1] == y(1) );
			HK_TEST( xx->val[i*4+2] == y(2) );
			HK_TEST( xx->val[i*4+3] == y(3) );
		}
	}
	{
		hkMxVector<M> x; x.template setSqrtInverse<HK_ACC_12_BIT,HK_SQRT_SET_ZERO>(v);
		const _MXVector<M>* xx = (const _MXVector<M>*)&(x.m_vec);
		for (int i=0; i<M; ++i)
		{
			hkVector4 w; w.set(hkReal(i*4+1),hkReal(i*4+2),hkReal(i*4+3),hkReal(i*4+4)); 
			hkVector4 y; y.template setSqrtInverse<HK_ACC_12_BIT,HK_SQRT_SET_ZERO>(w);
			HK_TEST( xx->val[i*4  ] == y(0) );
			HK_TEST( xx->val[i*4+1] == y(1) );
			HK_TEST( xx->val[i*4+2] == y(2) );
			HK_TEST( xx->val[i*4+3] == y(3) );
		}
	}
	{
		hkMxVector<M> x; x.template setSqrtInverse<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>(v);
		const _MXVector<M>* xx = (const _MXVector<M>*)&(x.m_vec);
		for (int i=0; i<M; ++i)
		{
			hkVector4 w; w.set(hkReal(i*4+1),hkReal(i*4+2),hkReal(i*4+3),hkReal(i*4+4)); 
			hkVector4 y; y.template setSqrtInverse<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>(w);
			HK_TEST( xx->val[i*4  ] == y(0) );
			HK_TEST( xx->val[i*4+1] == y(1) );
			HK_TEST( xx->val[i*4+2] == y(2) );
			HK_TEST( xx->val[i*4+3] == y(3) );
		}
	}
}

template <int M, int N>
static void mxvector_getSet()
{
	hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);
	hkMxVector<M> mone; mone.template setConstant<HK_QUADREAL_MINUS1>();
	hkVector4 w; w.set(hkReal(N*4),hkReal(N*4+1),hkReal(N*4+2),hkReal(N*4+3)); 

	hkVector4 x = v.template getVector<N>();
	HK_TEST(x(0) == w(0));
	HK_TEST(x(1) == w(1));
	HK_TEST(x(2) == w(2));
	HK_TEST(x(3) == w(3));

	w.setZero();
	v.template setVector<N>(w);
	v.add(mone);

	hkVector4 y;
	v.template getVector<N>(y);
	HK_TEST(y(0) == -1.0f);
	HK_TEST(y(1) == -1.0f);
	HK_TEST(y(2) == -1.0f);
	HK_TEST(y(3) == -1.0f);

	hkMxVector<M> z; z.template setScalarBroadcast<N>(v);
	const _MXVector<M>* zz = (const _MXVector<M>*)&(z.m_vec);
	for (int i=0; i<M; ++i)
	{
		if (i == N)
		{
			HK_TEST( zz->val[i*4] == -1.0f );
			HK_TEST( zz->val[i*4+1] == -1.0f );
			HK_TEST( zz->val[i*4+2] == -1.0f );
			HK_TEST( zz->val[i*4+3] == -1.0f );
		}
		else
		{
			HK_TEST( zz->val[i*4] == hkReal(i*4+N-1) );
			HK_TEST( zz->val[i*4+1] == hkReal(i*4+N-1) );
			HK_TEST( zz->val[i*4+2] == hkReal(i*4+N-1) );
			HK_TEST( zz->val[i*4+3] == hkReal(i*4+N-1) );
		}
	}
}

template <int M>
static void mxvector_gather()
{
	HK_ALIGN_REAL(static hkReal data[1024]);
	for (int i=0; i<1024; ++i) data[i] = hkReal(i);

	{
		hkMxVector<M> v; v.template gather<hkSizeOf(hkVector4)>((const hkVector4*)data);
		const _MXVector<M>* vv = (const _MXVector<M>*)&(v.m_vec);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(vv->val[i*4] == hkReal(i*4));
			HK_TEST(vv->val[i*4+1] == hkReal(i*4+1));
			HK_TEST(vv->val[i*4+2] == hkReal(i*4+2));
			HK_TEST(vv->val[i*4+3] == hkReal(i*4+3));
		}
	}

#ifndef MX_VECTOR_IS_AVX
	{
		hkMxVector<M> v; 
		const _MXVector<M>* vv = (const _MXVector<M>*)&(v.m_vec);

		v.setZero(); v.template gather<hkSizeOf(hkVector4), 0>((const hkVector4*)data);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(vv->val[i*4] == hkReal(0)); HK_TEST(vv->val[i*4+1] == hkReal(0)); HK_TEST(vv->val[i*4+2] == hkReal(0)); HK_TEST(vv->val[i*4+3] == hkReal(0));
		}

		v.setZero(); v.template gather<hkSizeOf(hkVector4), 1>((const hkVector4*)data);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(vv->val[i*4] == hkReal(i*4));
			HK_TEST(vv->val[i*4+1] == hkReal(0)); HK_TEST(vv->val[i*4+2] == hkReal(0)); HK_TEST(vv->val[i*4+3] == hkReal(0));
		}

		v.setZero(); v.template gather<hkSizeOf(hkVector4), 2>((const hkVector4*)data);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(vv->val[i*4] == hkReal(i*4));
			HK_TEST(vv->val[i*4+1] == hkReal(i*4+1));
			HK_TEST(vv->val[i*4+2] == hkReal(0)); HK_TEST(vv->val[i*4+3] == hkReal(0));
		}

		v.setZero(); v.template gather<hkSizeOf(hkVector4), 3>((const hkVector4*)data);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(vv->val[i*4] == hkReal(i*4));
			HK_TEST(vv->val[i*4+1] == hkReal(i*4+1));
			HK_TEST(vv->val[i*4+2] == hkReal(i*4+2));
			HK_TEST(vv->val[i*4+3] == hkReal(0));
		}

		v.setZero(); v.template gather<hkSizeOf(hkVector4), 4>((const hkVector4*)data);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(vv->val[i*4] == hkReal(i*4));
			HK_TEST(vv->val[i*4+1] == hkReal(i*4+1));
			HK_TEST(vv->val[i*4+2] == hkReal(i*4+2));
			HK_TEST(vv->val[i*4+3] == hkReal(i*4+3));
		}
	}
#endif
	{
		hkMxVector<M> v; v.template gather<2*hkSizeOf(hkVector4)>((const hkVector4*)data);
		const _MXVector<M>* vv = (const _MXVector<M>*)&(v.m_vec);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(vv->val[i*4] == hkReal(2*i*4));
			HK_TEST(vv->val[i*4+1] == hkReal(2*i*4+1));
			HK_TEST(vv->val[i*4+2] == hkReal(2*i*4+2));
			HK_TEST(vv->val[i*4+3] == hkReal(2*i*4+3));
		}
	}
	{
		hkArray<hkUint16> idx; for (int i=0; i<M; i++) idx.pushBack(hkUint16(i*2));
		hkMxVector<M> v; v.template gather<hkSizeOf(hkVector4)>((const hkVector4*)data, idx.begin());
		const _MXVector<M>* vv = (const _MXVector<M>*)&(v.m_vec);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(vv->val[i*4] == hkReal(idx[i]*4));
			HK_TEST(vv->val[i*4+1] == hkReal(idx[i]*4+1));
			HK_TEST(vv->val[i*4+2] == hkReal(idx[i]*4+2));
			HK_TEST(vv->val[i*4+3] == hkReal(idx[i]*4+3));
		}
	}
	{
		hkArray<hkInt32> idx; for (int i=0; i<M; i++) idx.pushBack(i+3);
		hkMxVector<M> v; v.template gather<hkSizeOf(hkVector4)>((const hkVector4*)data, idx.begin());
		const _MXVector<M>* vv = (const _MXVector<M>*)&(v.m_vec);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(vv->val[i*4] == hkReal(idx[i]*4));
			HK_TEST(vv->val[i*4+1] == hkReal(idx[i]*4+1));
			HK_TEST(vv->val[i*4+2] == hkReal(idx[i]*4+2));
			HK_TEST(vv->val[i*4+3] == hkReal(idx[i]*4+3));
		}
	}
	{
		const void* base[M]; for (int i=0; i<M; i++) base[i] = ((char*)(data+(3*i*4)))-3;
		hkMxVector<M> v; v.template gatherWithOffset<3>(base);
		const _MXVector<M>* vv = (const _MXVector<M>*)&(v.m_vec);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(vv->val[i*4] == hkReal(3*i*4));
			HK_TEST(vv->val[i*4+1] == hkReal(3*i*4+1));
			HK_TEST(vv->val[i*4+2] == hkReal(3*i*4+2));
			HK_TEST(vv->val[i*4+3] == hkReal(3*i*4+3));
		}
	}
#ifndef MX_VECTOR_IS_AVX
	{
		const void* base[M]; for (int i=0; i<M; i++) base[i] = ((char*)(data+(3*i*4)))-3;
		hkMxVector<M> v; 
		const _MXVector<M>* vv = (const _MXVector<M>*)&(v.m_vec);

		v.setZero(); v.template gatherWithOffset<3, 0>(base);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(vv->val[i*4] == hkReal(0)); HK_TEST(vv->val[i*4+1] == hkReal(0)); HK_TEST(vv->val[i*4+2] == hkReal(0)); HK_TEST(vv->val[i*4+3] == hkReal(0));
		}

		v.setZero(); v.template gatherWithOffset<3, 1>(base);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(vv->val[i*4] == hkReal(3*i*4));
			HK_TEST(vv->val[i*4+1] == hkReal(0)); HK_TEST(vv->val[i*4+2] == hkReal(0)); HK_TEST(vv->val[i*4+3] == hkReal(0));
		}

		v.setZero(); v.template gatherWithOffset<3, 2>(base);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(vv->val[i*4] == hkReal(3*i*4));
			HK_TEST(vv->val[i*4+1] == hkReal(3*i*4+1));
			HK_TEST(vv->val[i*4+2] == hkReal(0)); HK_TEST(vv->val[i*4+3] == hkReal(0));
		}

		v.setZero(); v.template gatherWithOffset<3, 3>(base);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(vv->val[i*4] == hkReal(3*i*4));
			HK_TEST(vv->val[i*4+1] == hkReal(3*i*4+1));
			HK_TEST(vv->val[i*4+2] == hkReal(3*i*4+2));
			HK_TEST(vv->val[i*4+3] == hkReal(0));
		}

		v.setZero(); v.template gatherWithOffset<3, 4>(base);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(vv->val[i*4] == hkReal(3*i*4));
			HK_TEST(vv->val[i*4+1] == hkReal(3*i*4+1));
			HK_TEST(vv->val[i*4+2] == hkReal(3*i*4+2));
			HK_TEST(vv->val[i*4+3] == hkReal(3*i*4+3));
		}
	}
#endif
}

template <int M>
static void mxvector_scatter()
{
	HK_ALIGN_REAL(static hkReal data[1024]);
	for (int i=0; i<1024; ++i) data[i] = -99.0f;

	{
		hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);
		v.template scatter<hkSizeOf(hkVector4)>((hkVector4*)data);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(data[i*4] == hkReal(i*4));
			HK_TEST(data[i*4+1] == hkReal(i*4+1));
			HK_TEST(data[i*4+2] == hkReal(i*4+2));
			HK_TEST(data[i*4+3] == hkReal(i*4+3));
		}
		for (int i=M*4; i<1024; ++i)
		{
			HK_TEST(data[i] == -99.0f);
		}
	}
#ifndef MX_VECTOR_IS_AVX
	for (int i=0; i<1024; ++i) data[i] = -99.0f;
	{
		hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);

		v.template scatter<hkSizeOf(hkVector4), 0>((hkVector4*)data);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(data[i*4] == -99.0f); HK_TEST(data[i*4+1] == -99.0f); HK_TEST(data[i*4+2] == -99.0f); HK_TEST(data[i*4+3] == -99.0f);
		}
		for (int i=M*4; i<1024; ++i)
		{
			HK_TEST(data[i] == -99.0f);
		}

		for (int i=0; i<1024; ++i) data[i] = -99.0f;
		v.template scatter<hkSizeOf(hkVector4), 1>((hkVector4*)data);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(data[i*4] == hkReal(i*4));
			HK_TEST(data[i*4+1] == -99.0f); HK_TEST(data[i*4+2] == -99.0f); HK_TEST(data[i*4+3] == -99.0f);
		}
		for (int i=M*4; i<1024; ++i)
		{
			HK_TEST(data[i] == -99.0f);
		}

		for (int i=0; i<1024; ++i) data[i] = -99.0f;
		v.template scatter<hkSizeOf(hkVector4), 2>((hkVector4*)data);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(data[i*4] == hkReal(i*4));
			HK_TEST(data[i*4+1] == hkReal(i*4+1));
			HK_TEST(data[i*4+2] == -99.0f); HK_TEST(data[i*4+3] == -99.0f);
		}
		for (int i=M*4; i<1024; ++i)
		{
			HK_TEST(data[i] == -99.0f);
		}

		for (int i=0; i<1024; ++i) data[i] = -99.0f;
		v.template scatter<hkSizeOf(hkVector4), 3>((hkVector4*)data);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(data[i*4] == hkReal(i*4));
			HK_TEST(data[i*4+1] == hkReal(i*4+1));
			HK_TEST(data[i*4+2] == hkReal(i*4+2));
			HK_TEST(data[i*4+3] == -99.0f);
		}
		for (int i=M*4; i<1024; ++i)
		{
			HK_TEST(data[i] == -99.0f);
		}

		for (int i=0; i<1024; ++i) data[i] = -99.0f;
		v.template scatter<hkSizeOf(hkVector4), 4>((hkVector4*)data);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(data[i*4] == hkReal(i*4));
			HK_TEST(data[i*4+1] == hkReal(i*4+1));
			HK_TEST(data[i*4+2] == hkReal(i*4+2));
			HK_TEST(data[i*4+3] == hkReal(i*4+3));
		}
		for (int i=M*4; i<1024; ++i)
		{
			HK_TEST(data[i] == -99.0f);
		}
	}
#endif
	for (int i=0; i<1024; ++i) data[i] = -99.0f;
	{
		hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);
		v.template scatter<2*sizeof(hkVector4)>((hkVector4*)data);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(data[2*i*4] == hkReal(i*4));
			HK_TEST(data[2*i*4+1] == hkReal(i*4+1));
			HK_TEST(data[2*i*4+2] == hkReal(i*4+2));
			HK_TEST(data[2*i*4+3] == hkReal(i*4+3));
			HK_TEST(data[2*i*4+4] == -99.0f);
			HK_TEST(data[2*i*4+5] == -99.0f);
			HK_TEST(data[2*i*4+6] == -99.0f);
			HK_TEST(data[2*i*4+7] == -99.0f);
		}
		for (int i=2*M*4; i<1024; ++i)
		{
			HK_TEST(data[i] == -99.0f);
		}
	}
	for (int i=0; i<1024; ++i) data[i] = -99.0f;
	{
		hkArray<hkUint16> idx; for (int i=0; i<M; i++) idx.pushBack(hkUint16(i*2));
		hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);
		v.template scatter<sizeof(hkVector4)>((hkVector4*)data, idx.begin());
		for (int i=0; i<M; ++i)
		{
			HK_TEST(data[2*i*4] == hkReal(i*4));
			HK_TEST(data[2*i*4+1] == hkReal(i*4+1));
			HK_TEST(data[2*i*4+2] == hkReal(i*4+2));
			HK_TEST(data[2*i*4+3] == hkReal(i*4+3));
			HK_TEST(data[2*i*4+4] == -99.0f);
			HK_TEST(data[2*i*4+5] == -99.0f);
			HK_TEST(data[2*i*4+6] == -99.0f);
			HK_TEST(data[2*i*4+7] == -99.0f);
		}
		for (int i=2*M*4; i<1024; ++i)
		{
			HK_TEST(data[i] == -99.0f);
		}
	}
	for (int i=0; i<1024; ++i) data[i] = -99.0f;
	{
		hkArray<hkInt32> idx; for (int i=0; i<M; i++) idx.pushBack(i*2+3);
		hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);
		v.template scatter<sizeof(hkVector4)>((hkVector4*)data, idx.begin());
		for (int i=0; i<M; ++i)
		{
			HK_TEST(data[idx[i]*4] == hkReal(i*4));
			HK_TEST(data[idx[i]*4+1] == hkReal(i*4+1));
			HK_TEST(data[idx[i]*4+2] == hkReal(i*4+2));
			HK_TEST(data[idx[i]*4+3] == hkReal(i*4+3));
		}
		for (int j=0; j<idx[0]; ++j)
		{
			HK_TEST(data[j*4] == -99.0f);
			HK_TEST(data[j*4+1] == -99.0f);
			HK_TEST(data[j*4+2] == -99.0f);
			HK_TEST(data[j*4+3] == -99.0f);
		}
		for (int i=1; i<M; ++i)
		{
			for (int j=idx[i-1]+1; j<idx[i]; ++j)
			{
				HK_TEST(data[j*4] == -99.0f);
				HK_TEST(data[j*4+1] == -99.0f);
				HK_TEST(data[j*4+2] == -99.0f);
				HK_TEST(data[j*4+3] == -99.0f);
			}
		}
		for (int j=4*(idx[M-1]+1); j<1024; ++j)
		{
			HK_TEST(data[j] == -99.0f);
		}
	}
	for (int i=0; i<1024; ++i) data[i] = -99.0f;
	{
		void* base[M]; for (int i=0; i<M; i++) base[i] = ((char*)(data+(3*i*4)))-3;
		hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);
		v.template scatterWithOffset<3>(base);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(data[3*i*4] == hkReal(i*4));
			HK_TEST(data[3*i*4+1] == hkReal(i*4+1));
			HK_TEST(data[3*i*4+2] == hkReal(i*4+2));
			HK_TEST(data[3*i*4+3] == hkReal(i*4+3));
			HK_TEST(data[3*i*4+4] == -99.0f);
			HK_TEST(data[3*i*4+5] == -99.0f);
			HK_TEST(data[3*i*4+6] == -99.0f);
			HK_TEST(data[3*i*4+7] == -99.0f);
			HK_TEST(data[3*i*4+8] == -99.0f);
			HK_TEST(data[3*i*4+9] == -99.0f);
			HK_TEST(data[3*i*4+10] == -99.0f);
			HK_TEST(data[3*i*4+11] == -99.0f);
		}
		for (int i=3*M*4; i<1024; ++i)
		{
			HK_TEST(data[i] == -99.0f);
		}
	}
#ifndef MX_VECTOR_IS_AVX
	for (int i=0; i<1024; ++i) data[i] = -99.0f;
	{
		void* base[M]; for (int i=0; i<M; i++) base[i] = ((char*)(data+(3*i*4)))-3;
		hkMxVector<M> v; v.template load<4>((const hkReal*)s_mxvector_aligned_data);

		v.template scatterWithOffset<3, 0>(base);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(data[3*i*4] == -99.0f); HK_TEST(data[3*i*4+1] == -99.0f); HK_TEST(data[3*i*4+2] == -99.0f); HK_TEST(data[3*i*4+3] == -99.0f);
			HK_TEST(data[3*i*4+4] == -99.0f);
			HK_TEST(data[3*i*4+5] == -99.0f);
			HK_TEST(data[3*i*4+6] == -99.0f);
			HK_TEST(data[3*i*4+7] == -99.0f);
			HK_TEST(data[3*i*4+8] == -99.0f);
			HK_TEST(data[3*i*4+9] == -99.0f);
			HK_TEST(data[3*i*4+10] == -99.0f);
			HK_TEST(data[3*i*4+11] == -99.0f);
		}
		for (int i=3*M*4; i<1024; ++i)
		{
			HK_TEST(data[i] == -99.0f);
		}

		for (int i=0; i<1024; ++i) data[i] = -99.0f;
		v.template scatterWithOffset<3, 1>(base);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(data[3*i*4] == hkReal(i*4));
			HK_TEST(data[3*i*4+1] == -99.0f); HK_TEST(data[3*i*4+2] == -99.0f); HK_TEST(data[3*i*4+3] == -99.0f);
			HK_TEST(data[3*i*4+4] == -99.0f);
			HK_TEST(data[3*i*4+5] == -99.0f);
			HK_TEST(data[3*i*4+6] == -99.0f);
			HK_TEST(data[3*i*4+7] == -99.0f);
			HK_TEST(data[3*i*4+8] == -99.0f);
			HK_TEST(data[3*i*4+9] == -99.0f);
			HK_TEST(data[3*i*4+10] == -99.0f);
			HK_TEST(data[3*i*4+11] == -99.0f);
		}
		for (int i=3*M*4; i<1024; ++i)
		{
			HK_TEST(data[i] == -99.0f);
		}

		for (int i=0; i<1024; ++i) data[i] = -99.0f;
		v.template scatterWithOffset<3, 2>(base);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(data[3*i*4] == hkReal(i*4));
			HK_TEST(data[3*i*4+1] == hkReal(i*4+1));
			HK_TEST(data[3*i*4+2] == -99.0f); HK_TEST(data[3*i*4+3] == -99.0f);
			HK_TEST(data[3*i*4+4] == -99.0f);
			HK_TEST(data[3*i*4+5] == -99.0f);
			HK_TEST(data[3*i*4+6] == -99.0f);
			HK_TEST(data[3*i*4+7] == -99.0f);
			HK_TEST(data[3*i*4+8] == -99.0f);
			HK_TEST(data[3*i*4+9] == -99.0f);
			HK_TEST(data[3*i*4+10] == -99.0f);
			HK_TEST(data[3*i*4+11] == -99.0f);
		}
		for (int i=3*M*4; i<1024; ++i)
		{
			HK_TEST(data[i] == -99.0f);
		}

		for (int i=0; i<1024; ++i) data[i] = -99.0f;
		v.template scatterWithOffset<3, 3>(base);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(data[3*i*4] == hkReal(i*4));
			HK_TEST(data[3*i*4+1] == hkReal(i*4+1));
			HK_TEST(data[3*i*4+2] == hkReal(i*4+2));
			HK_TEST(data[3*i*4+3] == -99.0f);
			HK_TEST(data[3*i*4+4] == -99.0f);
			HK_TEST(data[3*i*4+5] == -99.0f);
			HK_TEST(data[3*i*4+6] == -99.0f);
			HK_TEST(data[3*i*4+7] == -99.0f);
			HK_TEST(data[3*i*4+8] == -99.0f);
			HK_TEST(data[3*i*4+9] == -99.0f);
			HK_TEST(data[3*i*4+10] == -99.0f);
			HK_TEST(data[3*i*4+11] == -99.0f);
		}
		for (int i=3*M*4; i<1024; ++i)
		{
			HK_TEST(data[i] == -99.0f);
		}

		for (int i=0; i<1024; ++i) data[i] = -99.0f;
		v.template scatterWithOffset<3, 4>(base);
		for (int i=0; i<M; ++i)
		{
			HK_TEST(data[3*i*4] == hkReal(i*4));
			HK_TEST(data[3*i*4+1] == hkReal(i*4+1));
			HK_TEST(data[3*i*4+2] == hkReal(i*4+2));
			HK_TEST(data[3*i*4+3] == hkReal(i*4+3));
			HK_TEST(data[3*i*4+4] == -99.0f);
			HK_TEST(data[3*i*4+5] == -99.0f);
			HK_TEST(data[3*i*4+6] == -99.0f);
			HK_TEST(data[3*i*4+7] == -99.0f);
			HK_TEST(data[3*i*4+8] == -99.0f);
			HK_TEST(data[3*i*4+9] == -99.0f);
			HK_TEST(data[3*i*4+10] == -99.0f);
			HK_TEST(data[3*i*4+11] == -99.0f);
		}
		for (int i=3*M*4; i<1024; ++i)
		{
			HK_TEST(data[i] == -99.0f);
		}
	}
#endif
}

template <int M>
static void mxvector_Tests()
{
	mxvector_load_assign<M>();
	mxvector_store_assign<M>();
	// 	mxvector_setPermutation<M>();
	mxvector_horizontal<M,4>();
	mxvector_horizontal<M,3>();
	mxvector_setAsBC<M>();
	mxvector_setAddMul<M>();
#if !defined(HK_PLATFORM_LINUX)
	mxvector_storeTransposed4<M>();
#endif
	mxvector_isOk<M,4>();
	mxvector_isOk<M,3>();
	mxvector_isOk<M,2>();
	mxvector_isOk<M,1>();
	mxvector_reduceAdd<M>();
	mxvector_dot<M,4>();
	mxvector_dot<M,3>();
	mxvector_dot<M,2>();
	mxvector_length<M,4>();
	mxvector_length<M,3>();
	mxvector_length<M,2>();
	mxvector_constant<M,HK_QUADREAL_MINUS1>();
	mxvector_constant<M,HK_QUADREAL_1>();
	mxvector_constant<M,HK_QUADREAL_EPS>();
	mxvector_select<M>();
	mxvector_compare<M>();
	mxvector_setInterpolate<M>();
	mxvector_minMaxNegAbs<M>();
	mxvector_cross<M>();
	mxvector_calc<M>();
	mxvector_xyzw<M>();
	mxvector_normalize<M,4>();
	mxvector_normalize<M,3>();
	mxvector_sqrt<M>();
	mxvector_getSet<M,3%M>();
	mxvector_getSet<M,2%M>();
	mxvector_getSet<M,1%M>();
	mxvector_getSet<M,0%M>();
	mxvector_gather<M>();
	mxvector_scatter<M>();
}

int mxvector_main()
{
	for (int i=0; i<512; i++) s_mxvector_byte_data[i] = hkUint8( i%16 );

	mxvector_Tests<4>();
	mxvector_Tests<3>();
	mxvector_Tests<2>();
	mxvector_Tests<1>();

	return 0;
}


// mxvector.cpp test assumes 'precise' float control,
// and can fail if model is set to 'fast'. (See issue COM-1779)
#ifdef HK_PLATFORM_WIN32
#if (HK_CONFIG_SIMD==HK_CONFIG_SIMD_DISABLED)
#pragma float_control(except, off) 
#pragma float_control(precise, off) 
#endif
#endif


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(mxvector_main, "Fast", "Common/Test/UnitTest/Base/", "UnitTest/Math/Linear/mxvector.cpp"     );

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
