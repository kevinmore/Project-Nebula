/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>

#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Common/Base/Math/Vector/hkIntVector.h>

namespace
{
	union hkVectorUintUnion
	{
		hkUint32 u[4];
		hkIntVector v;
	};

	union hkVectorShortUnion
	{
		hkUint16 u[8];
		hkIntVector v;
	};

	union hkVectorByteUnion
	{
		hkUint8 u[16];
		hkIntVector v;
	};
}

static void loadAndSet()
{
	// aligned load
	{
		HK_ALIGN16(char buf[32]);
		for (int i=0; i<32; ++i) buf[i] = (char)i;

		hkIntVector v; v.setZero();
#if !defined(HK_ALIGN_RELAX_CHECKS)
		HK_TEST_ASSERT(0x70aae483, v.load<4>((hkUint32*)(buf+1)));
		HK_TEST_ASSERT(0x70aae483, v.load<3>((hkUint32*)(buf+1)));
		HK_TEST_ASSERT(0x70aae483, v.load<2>((hkUint32*)(buf+1)));
#if !defined(HK_COMPILER_SNC)
		HK_TEST_ASSERT(0x70aae483, v.load<1>((hkUint32*)(buf+1)));
#endif
		HK_TEST_ASSERT(0x70aae483, v.load<4>((hkUint32*)(buf+4)));
		HK_TEST_ASSERT(0x70aae483, v.load<3>((hkUint32*)(buf+4)));
		HK_TEST_ASSERT(0x70aae483, v.load<2>((hkUint32*)(buf+4)));
#endif
		v.load<1>((hkUint32*)(buf+4));
		for (int i=0; i<4; ++i)
		{
			HK_TEST( ((char*)&(v.m_quad))[i] == buf[4+i] );
		}

#if !defined(HK_ALIGN_RELAX_CHECKS)
#if (HK_NATIVE_ALIGN_CHECK == 0xF) // Loading from 8-byte alignment
		HK_TEST_ASSERT(0x70aae483, v.load<4>((hkUint32*)(buf+8)));
#endif
		HK_TEST_ASSERT(0x70aae483, v.load<3>((hkUint32*)(buf+8)));
#endif
		v.load<2>((hkUint32*)(buf+8));
		for (int i=0; i<8; ++i)
		{
			HK_TEST( ((char*)&(v.m_quad))[i] == buf[8+i] );
		}

		v.load<3>((hkUint32*)(buf+16));
		for (int i=0; i<12; ++i)
		{
			HK_TEST( ((char*)&(v.m_quad))[i] == buf[16+i] );
		}

		v.load<4>((hkUint32*)(buf));
		for (int i=0; i<16; ++i)
		{
			HK_TEST( ((char*)&(v.m_quad))[i] == buf[i] );
		}
	}

	// unaligned load
	hkUint32 array[] = {0,1,2,3,4,5,6};
	for (int j=0; j<4; j++)
	{
		hkUint32* p = &array[j];

		hkVectorUintUnion x;
		x.v.load<4, HK_IO_NATIVE_ALIGNED>(p);

		for (int i=0; i<4; i++)
		{
			HK_TEST2(x.u[i] == p[i], "hkIntVector::unalignedloadUnaligned4");
		}

		x.v.setZero();
		for (int i=0; i<4; i++)
		{
			HK_TEST2(x.u[i] == 0, "hkIntVector::setZero");
		}
	}

	// test setImmediate*
	{
		hkVectorUintUnion x;
		x.v.splatImmediate32<3>();
		HK_TEST(x.u[0] == 3);
		HK_TEST(x.u[1] == 3);
		HK_TEST(x.u[2] == 3);
		HK_TEST(x.u[3] == 3);

		hkVectorShortUnion y;
		y.v.splatImmediate16<4>();
		for (int i=0; i<8; i++)
		{
			HK_TEST(y.u[i] == 4);
		}

		hkVectorByteUnion z;
		z.v.splatImmediate8<5>();
		for (int i=0; i<16; i++)
		{
			HK_TEST(z.u[i] == 5);
		}
	}

	// test setBroadcast
	{
		hkVectorUintUnion x, y0, y1, y2, y3;
		x.v.load<4, HK_IO_NATIVE_ALIGNED>(array+3);

		y0.v.setBroadcast<0>(x.v);
		y1.v.setBroadcast<1>(x.v);
		y2.v.setBroadcast<2>(x.v);
		y3.v.setBroadcast<3>(x.v);

		for (int i=0; i<4; i++)	{ HK_TEST(y0.u[i] == x.u[0]); }
		for (int i=0; i<4; i++)	{ HK_TEST(y1.u[i] == x.u[1]); }
		for (int i=0; i<4; i++)	{ HK_TEST(y2.u[i] == x.u[2]); }
		for (int i=0; i<4; i++)	{ HK_TEST(y3.u[i] == x.u[3]); }
	}
}

static void binaryOperations()
{
	hkUint32 a[] = {128374, 234867, 192768, 2374687};
	hkUint32 b[] = {19236, 9728635, 71656,  126};

	hkVectorUintUnion c, d, resultAnd, resultOr, resultAndC, resultXor, resultNot;
	c.v.load<4, HK_IO_NATIVE_ALIGNED>(a);
	d.v.load<4, HK_IO_NATIVE_ALIGNED>(b);
	
	resultAnd.v.setAnd(c.v, d.v);
	resultOr.v.setOr(c.v, d.v);
	resultAndC.v.setAndNot(c.v, d.v);
	resultXor.v.setXor(c.v, d.v);
	
	resultNot.v.setNot(c.v);
	

	for(int i=0; i<4; i++)
	{
		HK_TEST(resultAnd.u[i]  == (a[i] & b[i]) );
		HK_TEST(resultOr.u[i]   == (a[i] | b[i]) );
		HK_TEST(resultAndC.u[i] == (a[i] & ~b[i]) );
		HK_TEST(resultXor.u[i]  == (a[i] ^ b[i]) );

		HK_TEST(resultNot.u[i] ==  ~a[i] );
	}

}

static void arithmetic()
{
	// Pick these so that at least one overflows and one doesn't
	hkUint32 a32[] = {0xFFFFFFFE, 0xFFFFFFFE, 0xFFFFFFFE, 0xFFFFFFFD};
	hkUint32 b32[] = {0,1,2,1};

	hkVectorUintUnion a, b;
	a.v.load<4, HK_IO_NATIVE_ALIGNED>(a32);
	b.v.load<4, HK_IO_NATIVE_ALIGNED>(b32);

	{
		hkVectorUintUnion result; 
		result.v.setAddSaturateU32(a.v, b.v);
		HK_TEST(result.u[0] == 0xFFFFFFFE);
		HK_TEST(result.u[1] == 0xFFFFFFFF);
		HK_TEST(result.u[2] == 0xFFFFFFFF);
		HK_TEST(result.u[3] == 0xFFFFFFFE);

		result.v.setAddSaturateU32(a.v, b.v);
		HK_TEST(result.u[0] == 0xFFFFFFFE);
		HK_TEST(result.u[1] == 0xFFFFFFFF);
		HK_TEST(result.u[2] == 0xFFFFFFFF);
		HK_TEST(result.u[3] == 0xFFFFFFFE);
	}

	HK_ALIGN16(hkUint16 c16[]) = {0xFFFE, 0xFFFE, 0xFFFE, 0xFFFD,    4,5,6,7};
	HK_ALIGN16(hkUint16 d16[]) = {0,1,2,1,  12,13,14,15};

	hkVectorUintUnion c, d;
	c.v.load<4, HK_IO_NATIVE_ALIGNED>((hkUint32*) c16 );
	d.v.load<4, HK_IO_NATIVE_ALIGNED>((hkUint32*) d16 );

	{
		hkVectorShortUnion result;
		result.v.setAddSaturateU16(c.v, d.v);
		HK_TEST(result.u[0] == 0xFFFE);
		HK_TEST(result.u[1] == 0xFFFF);
		HK_TEST(result.u[2] == 0xFFFF);
		HK_TEST(result.u[3] == 0xFFFE);
		for (int i=4; i<8; i++)
		{
			HK_TEST(result.u[i] == c16[i] + d16[i]);
		}
	}


	// Subtraction
	hkUint32 e32[] = {10, 20, 20, 20};
	hkUint32 f32[] = {20, 10, 19, 20};
	hkVectorUintUnion e, f;
	e.v.load<4, HK_IO_NATIVE_ALIGNED>(e32);
	f.v.load<4, HK_IO_NATIVE_ALIGNED>(f32);
	{
		hkVectorUintUnion result; 
		result.v.setSubSaturateU32(e.v, f.v);
		HK_TEST(result.u[0] == 0);
		HK_TEST(result.u[1] == 10);
		HK_TEST(result.u[2] == 1);
		HK_TEST(result.u[3] == 0);
	}
	{
		// test big-small and small-big (in case of any signed/unsigned mishaps
		hkVectorUintUnion bigMinusSmall, smallMinusBig; 
		bigMinusSmall.v.setSubSaturateU32(a.v, e.v);
		smallMinusBig.v.setSubSaturateU32(e.v, a.v);
		for (int i=0; i<4; i++)
		{
			HK_TEST(bigMinusSmall.u[i] == a32[i] - e32[i]);
			HK_TEST(smallMinusBig.u[i] == 0);
		}
	}

	HK_ALIGN16(hkUint16 g16[]) = {10, 20, 20, 20,  0xFFFF, 13,     0xFFFF, 7 };
	HK_ALIGN16(hkUint16 h16[]) = {20, 10, 19, 20,  12,     0xFFFF, 0xFFFF, 8 };
	hkVectorUintUnion g,h;
	g.v.load<4, HK_IO_NATIVE_ALIGNED>((hkUint32*) g16 );
	h.v.load<4, HK_IO_NATIVE_ALIGNED>((hkUint32*) h16 );

	{
		hkVectorShortUnion result;
		result.v.setSubSaturateU16(g.v, h.v);
		HK_TEST(result.u[0] == 0);
		HK_TEST(result.u[1] == 10);
		HK_TEST(result.u[2] == 1);
		HK_TEST(result.u[3] == 0);
		HK_TEST(result.u[4] == 0xFFFF-12);
		HK_TEST(result.u[5] == 0);
		HK_TEST(result.u[6] == 0);
		HK_TEST(result.u[7] == 0);
	}

	// Signed 16 bit add / sub
	{
		HK_ALIGN16(hkQuadShortUnion A);
		HK_ALIGN16(hkQuadShortUnion B);
		HK_ALIGN16(hkQuadShortUnion ApB);	// A + B
		HK_ALIGN16(hkQuadShortUnion AmB);	// A - B

		A.u[0] = 32767;		A.u[1] = -32768;	A.u[2] = -32768;	A.u[3] = 3;		A.u[4] = 4;		A.u[5] = 5;		A.u[6] = 6;		A.u[7] = 7;
		B.u[0] = 32767;		B.u[1] = -32768;	B.u[2] = +32767;	B.u[3] = 8;		B.u[4] = 9;		B.u[5] = 10;	B.u[6] = 11;	B.u[7] = 12;
		ApB.u[0] = 32767;	ApB.u[1] = -32768;	ApB.u[2] = -1;		ApB.u[3] = 11;	ApB.u[4] = 13;	ApB.u[5] = 15;	ApB.u[6] = 17;	ApB.u[7] = 19;
		AmB.u[0] = 0;		AmB.u[1] = 0;		AmB.u[2] = -32768;	AmB.u[3] = -5;	AmB.u[4] = -5;	AmB.u[5] = -5;	AmB.u[6] = -5;	AmB.u[7] = -5;

		hkIntVector vA, vB, vC; vC.setZero();
		vA.load<4>((hkUint32*)&A.q);
		vB.load<4>((hkUint32*)&B.q);

		vC.setAddSaturateS16(vA, vB);
		A.q = vC.m_quad;

		vC.setSubSaturateS16(vA, vB);
		B.q = vC.m_quad;

		for(int i = 0 ; i < 8 ; i++)
		{
			HK_TEST(A.u[i] == ApB.u[i]);
			HK_TEST(B.u[i] == AmB.u[i]);
		}
	}

	// Multiplication & dot products
	{
		hkIntVector vA;		vA.set(2, 3, 4, -5);
		hkIntVector vB;		vB.set(6, -7, 8, 9);
		hkIntVector vAB;	vAB.setMul(vA, vB);

		HK_TEST(vAB.getComponent<0>() == 12);
		HK_TEST(vAB.getComponent<1>() == -21);
		HK_TEST(vAB.getComponent<2>() == 32);
		HK_TEST(vAB.getComponent<3>() == -45);

		const hkInt64 d1 = vA.dot<1>(vB);
		const hkInt64 d2 = vA.dot<2>(vB);
		const hkInt64 d3 = vA.dot<3>(vB);
		const hkInt64 d4 = vA.dot<4>(vB);
		HK_TEST(d1 == 12);
		HK_TEST(d2 == -9);
		HK_TEST(d3 == 23);
		HK_TEST(d4 == -22);
	}
}

static void shift()
{

	hkUint32 a32[] = {0x12345678, 0x23456789, 0x3456789a, 0x456789ab };
	hkUint32 b32[] = {1, 2, 3, 4};

	hkVectorUintUnion c,d, result;
	c.v.load<4, HK_IO_NATIVE_ALIGNED>(a32);
	d.v.load<4, HK_IO_NATIVE_ALIGNED>(b32);
	
	{
		// Right shift, constant
		result.v.setShiftRight32<18>(c.v);
		HK_TEST(result.u[0] == a32[0]>>18);
		HK_TEST(result.u[1] == a32[1]>>18);
		HK_TEST(result.u[2] == a32[2]>>18);
		HK_TEST(result.u[3] == a32[3]>>18);
	}

	{
		// Left shift, constant
		result.v.setShiftLeft32<19>(c.v);
		HK_TEST(result.u[0] == a32[0]<<19);
		HK_TEST(result.u[1] == a32[1]<<19);
		HK_TEST(result.u[2] == a32[2]<<19);
		HK_TEST(result.u[3] == a32[3]<<19);
	}

	{
		// Right shift, variable
		result.v.setShiftRight32(c.v, d.v);
		HK_TEST(result.u[0] == a32[0]>>1);
		HK_TEST(result.u[1] == a32[1]>>2);
		HK_TEST(result.u[2] == a32[2]>>3);
		HK_TEST(result.u[3] == a32[3]>>4);
	}

	{
		// Left shift, variable
		result.v.setShiftLeft32(c.v, d.v);
		HK_TEST(result.u[0] == a32[0]<<1);
		HK_TEST(result.u[1] == a32[1]<<2);
		HK_TEST(result.u[2] == a32[2]<<3);
		HK_TEST(result.u[3] == a32[3]<<4);
	}

	// 16 bit shifts
	{
		HK_ALIGN16(hkQuadShortUnion A);
		HK_ALIGN16(hkQuadShortUnion B);

		const int shift = 5;
		{
			// Init A = {1, 2, 3, 4, 5, 6, 7, 8}
			for(int k = 0 ; k < 8 ; k++)
			{
				A.u[k] = (hkInt16)(k + 1);
				B.u[k] = (hkInt16)(A.u[k] << shift);
			}

			// Shift A left
			hkIntVector vA;
			vA.load<4>((hkUint32*)&A.q);
			vA.setShiftLeft16<shift>(vA);
			A.q = vA.m_quad;

			for(int k = 0 ; k < 8 ; k++)
			{
				HK_TEST(A.u[k] == B.u[k]);
			}
		}
	}

	//128 bit shifts
	{
		HK_ALIGN16(hkQuadUcharUnion A);
		HK_ALIGN16(hkQuadUcharUnion B);
		HK_ALIGN16(hkQuadUcharUnion C);

		// Init A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
		const int shift = 5;
		for(int k = 0 ; k < 16 ; k++)
		{
			A.u[k] = (hkInt8)(k + 1);
		}
		for(int k = 0 ; k < 16 ; k++)
		{
			// Shift right response
			B.u[k] = (k < shift) ? 0 : A.u[k - shift];

			// Shift left response
			C.u[k] = (k > 15 - shift) ? 0 : A.u[k + shift];
		}

		// Shift A left
		hkIntVector vA, vB, vC; vC.setZero();
		vA.load<4>((hkUint32*)&A.q);

		vB.setShiftRight128<shift>(vA);
		vC.setShiftLeft128<shift>(vA);
		vC.setShiftLeft128<shift>(vA);

		A.q = vB.m_quad;
		for(int k = 0 ; k < 16 ; k++)
		{
			HK_TEST(A.u[k] == B.u[k]);
		}

		A.q = vC.m_quad;
		for(int k = 0 ; k < 16 ; k++)
		{
			HK_TEST(A.u[k] == C.u[k]);
		}
	}

}

static void merge_pack()
{


	{
		hkUint32 a32[] = {0,1,2,3};
		hkUint32 b32[] = {4,5,6,7};

		hkVectorUintUnion a, b;
		a.v.load<4, HK_IO_NATIVE_ALIGNED>(a32);
		b.v.load<4, HK_IO_NATIVE_ALIGNED>(b32);
		{
			hkVectorUintUnion result; 
			result.v.setMergeHead32(a.v, b.v);
			HK_TEST(result.u[0] == 0);
			HK_TEST(result.u[1] == 4);
			HK_TEST(result.u[2] == 1);
			HK_TEST(result.u[3] == 5);
		}
		{
			hkVectorUintUnion result; 
			result.v.setMergeTail32(a.v, b.v);
			HK_TEST(result.u[0] == 2);
			HK_TEST(result.u[1] == 6);
			HK_TEST(result.u[2] == 3);
			HK_TEST(result.u[3] == 7);
		}
	}
	
	{
		HK_ALIGN16(hkUint16 a16[]) = {0,1,2,3,    4,5,6,7};
		HK_ALIGN16(hkUint16 b16[]) = {8,9,10,11,  12,13,14,15};

		hkVectorUintUnion a, b;
		a.v.load<4, HK_IO_NATIVE_ALIGNED>( (hkUint32*)a16 );
		b.v.load<4, HK_IO_NATIVE_ALIGNED>( (hkUint32*)b16 );

		{
			hkVectorShortUnion resultHi, resultLo; 
			resultHi.v.setMergeHead16(a.v, b.v);	// resultHi should end up as {0,8,1,9,2,10,3,11};
			resultLo.v.setMergeTail16(a.v, b.v);	// resultLo should end up as (4,12,5,13,6,14,7,15}

			for (int i=0; i<4; i++)
			{
				HK_TEST(a16[i] == resultHi.u[2*i]);
				HK_TEST(b16[i] == resultHi.u[2*i + 1]);

				HK_TEST(a16[i+4] == resultLo.u[2*i]);
				HK_TEST(b16[i+4] == resultLo.u[2*i + 1]);
			}
		}
		{
			hkVectorShortUnion resultHi, resultLo; 
			resultHi.v.setCombineHead16To32(a.v, b.v);	// resultHi should end up as 0x00000008, 0x00010009, 0x0002000A, 0x0003000B
			resultLo.v.setCombineTail16To32(a.v, b.v);	// resultLo should end up as 0x0004000C, 0x0005000D, 0x0006000E, 0x0007000F

			hkVectorShortUnion packResult;
			packResult.v.setConvertU32ToU16(resultHi.v, resultLo.v);

			// packResult now should be equal to b;
			for (int i=0; i<8; i++)
			{
				HK_TEST(packResult.u[i] == b16[i]);
			}
		}
	}



	{
		// check sign correctness of setConvertU32ToU16
		HK_ALIGN16(hkInt32 aa16[]) = { 0x7fff, 0x7fff, 0x7fff, 0x7fff};
		HK_ALIGN16(hkInt32 ab16[]) = {-0x8000,-0x8000,-0x8000,-0x8000};

		hkVectorShortUnion packResult;
		packResult.v.setConvertU32ToU16( (const hkIntVector&) aa16, (const hkIntVector&)ab16);

		// packResult now should be equal to b;
		for (int i=0; i<4; i+=1)
		{
			HK_TEST(packResult.u[i]   == 0x7fff);
			HK_TEST(packResult.u[i+4] == 0x8000);
		}
	}


	{
		HK_ALIGN16(hkUint8 a8[16]);
		HK_ALIGN16(hkUint8 b8[16]);
		for (hkUint8 i=0; i<16; i++){		a8[i] = i;		b8[i] = i+16;	}

		hkVectorUintUnion c, d;
		c.v.load<4, HK_IO_NATIVE_ALIGNED>((hkUint32*) a8 );
		d.v.load<4, HK_IO_NATIVE_ALIGNED>((hkUint32*) b8 );

		{
			hkVectorByteUnion resultHi, resultLo; 
			resultHi.v.setMergeHead8(c.v, d.v);
			resultLo.v.setMergeTail8(c.v, d.v);
			for (int i=0; i<8; i++)
			{
				HK_TEST(a8[i] == resultHi.u[2*i]);
				HK_TEST(b8[i] == resultHi.u[2*i + 1]);

				HK_TEST(a8[i+8] == resultLo.u[2*i]);
				HK_TEST(b8[i+8] == resultLo.u[2*i + 1]);
			}
		}
	}

	// Pack signed unsigned saturate
	{
		HK_ALIGN16(hkQuadShortUnion A1);
		HK_ALIGN16(hkQuadShortUnion A2);
		HK_ALIGN16(hkQuadUcharUnion B1);
		HK_ALIGN16(hkQuadUcharUnion B2);

		for(int k = 0 ; k < 8 ; k++)
		{
			A1.u[k]		= (hkInt16)(k + 1);
			A2.u[k]		= (hkInt16)(k + 8);
			B2.u[k]		= (hkUint8)A1.u[k];
			B2.u[k + 8]	= (hkUint8)A2.u[k]; 
		}

		A1.u[0] = 256;
		A2.u[0] = -256;
		B2.u[0] = 255;
		B2.u[8] = 0;

		hkIntVector vA1, vA2, vA;
		vA1.load<4>((hkUint32*)&A1.q);
		vA2.load<4>((hkUint32*)&A2.q);

		vA.setConvertSaturateS16ToU8(vA1, vA2);
		B1.q = vA.m_quad;

		for(int k = 0 ; k < 16 ; k++)
		{
			HK_TEST(B1.u[k] == B2.u[k]);
		}
	}
}

static void select()
{
	hkUint32 a[] = {1,2,3,4};
	hkUint32 b[] = {5,6,7,8};

	hkVectorUintUnion x,y, z;
	x.v.load<4, HK_IO_NATIVE_ALIGNED>(a);
	y.v.load<4, HK_IO_NATIVE_ALIGNED>(b);

	int bitMasks[4] = {hkVector4ComparisonMask::MASK_X, hkVector4ComparisonMask::MASK_Y, hkVector4ComparisonMask::MASK_Z, hkVector4ComparisonMask::MASK_W};

	for (int m=0; m<16; m++)
	{
		hkVector4Comparison comp; comp.set((hkVector4Comparison::Mask) m);
		z.v.setSelect(comp, y.v, x.v);
		for (int i=0; i<4; i++)
		{
			HK_TEST( z.u[i] == ( (m & bitMasks[i]) ? y.u[i] : x.u[i] ) );
		}
	}
}

static void convertI2F()
{
	hkVector4 result;
	hkVector4 comparison;
	hkSimdReal eps;
	eps.setFromFloat(1e-3f);

	{
		// test uint->float
		hkUint32 a[] = {1,2,3,4};
		hkIntVector v;
		v.load<4, HK_IO_NATIVE_ALIGNED>(a);
		v.convertU32ToF32(result);

		comparison.set(1.0f, 2.0f, 3.0f, 4.0f);

		HK_TEST( result.allEqual<4>(comparison,eps) );
	}

	{
		// test int->float
#if defined(HK_PLATFORM_X64) && defined(HK_DEBUG) && defined(_MSC_VER) && (_MSC_VER==1500)
		// This is a workaround to avoid a VS2008 bug that happens in
		// debug mode. In this particular case the optimizer seems to drop 
		// the vector loading and using 'c' directly as source for the conversion
		// sse2 instruction.
		HK_ALIGN16(hkInt32 c[]) = {-1, 2, -3, 4};
		hkIntVector v;
		v.load<4>((hkUint32*) c);
#else
		hkInt32 c[] = {-1, 2, -3, 4};
		hkIntVector v;
		v.load<4, HK_IO_NATIVE_ALIGNED>((hkUint32*) c);
#endif
		v.convertS32ToF32(result);

		comparison.set(-1.0f, 2.0f, -3.0f, 4.0f);

		HK_TEST( result.allEqual<4>(comparison,eps) );
	}
}

static void shuffle()
{
	hkUint32 a32[] = {98836870, 25049423, 87906930, 75667782};

	hkVectorUintUnion x, result0, result1, result2, result3;
	x.v.load<4, HK_IO_NATIVE_ALIGNED>(a32);

	result0.v.setPermutation<hkVectorPermutation::YZWX>(x.v);
	result1.v.setPermutation<hkVectorPermutation::WXYZ>(x.v);
	result2.v.setPermutation<hkVectorPermutation::ZZZZ>(x.v);
	result3.v.setPermutation<hkVectorPermutation::XYZW>(x.v);

	for (int i=0; i<4; i++){ HK_TEST(result0.u[i] == a32[(i+1)%4]); }
	for (int i=0; i<4; i++){ HK_TEST(result1.u[i] == a32[(i+3)%4]); }
	for (int i=0; i<4; i++){ HK_TEST(result2.u[i] == a32[2]); }
	for (int i=0; i<4; i++){ HK_TEST(result3.u[i] == a32[i]); }
}

static void permute()
{
	static HK_ALIGN16(hkUint8 p[16]) = { 0x06, 0x03, 0x0a, 0x0e, 0x0f, 0x0b, 0x09, 0x05, 0x04, 0x00, 0x01, 0x0d, 0x08, 0x07, 0x02, 0x0c };
	static HK_ALIGN16(hkUint8 s[16]) = { 0x06, 0x13, 0x0a, 0x1e, 0x1f, 0x0b, 0x09, 0x15, 0x04, 0x10, 0x01, 0x1d, 0x18, 0x17, 0x02, 0x0c };
	hkUint32 a32[4] = {98836870, 25049423, 87906930, 75667782};
	hkUint32 b32[4] = {27429291, 93214205, 12315456, 93598211};

	{
		hkIntVector a; a.load<4, HK_IO_NATIVE_ALIGNED>(a32);
		hkIntVector perm; perm.setPermuteU8(a, *(hkIntVector*)p);
		hkUint8* pperm = (hkUint8*)&perm;
		hkUint8* pa32 = (hkUint8*)a32;
		for (int i=0; i<16; ++i)
		{
			HK_TEST( pperm[i] == pa32[p[i]] );
		}
	}

	{
		hkIntVector a; a.load<4, HK_IO_NATIVE_ALIGNED>(a32);
		hkIntVector perm; perm.setZero();
		HK_TEST_ASSERT(0xf820d0c2, perm.setPermuteU8(a, *(hkIntVector*)s));
	}

	{
		hkIntVector a; a.load<4, HK_IO_NATIVE_ALIGNED>(a32);
		hkIntVector b; b.load<4, HK_IO_NATIVE_ALIGNED>(b32);
		hkIntVector perm; perm.setPermuteU8(a, b, *(hkIntVector*)p);
		hkUint8* pperm = (hkUint8*)&perm;
		hkUint8* pa32 = (hkUint8*)a32;
		for (int i=0; i<16; ++i)
		{
			HK_TEST( pperm[i] == pa32[p[i]] );
		}
	}

	{
		hkIntVector a; a.load<4, HK_IO_NATIVE_ALIGNED>(a32);
		hkIntVector b; b.load<4, HK_IO_NATIVE_ALIGNED>(b32);
		hkIntVector perm; perm.setPermuteU8(a, b, *(hkIntVector*)s);
		hkUint8* pperm = (hkUint8*)&perm;
		hkUint8* pa32 = (hkUint8*)a32;
		hkUint8* pb32 = (hkUint8*)b32;
		for (int i=0; i<16; ++i)
		{
			if (s[i] & 0xf0)
			{
				HK_TEST( pperm[i] == pb32[s[i] & 0x0f] );
			}
			else
			{
				HK_TEST( pperm[i] == pa32[s[i] & 0x0f] );
			}
		}
	}
}

static void store()
{
	HK_ALIGN16(char buf[34]);
	for (int i=0; i<34; ++i) buf[i] = 99;

	// Store aligned
	{
		hkIntVector v; v.set(1, 2, 3, 4);
		hkUint32* ibuf = (hkUint32*)(buf+1);

#if !defined(HK_ALIGN_RELAX_CHECKS)
		HK_TEST_ASSERT(0x70aae483, v.store<4>(ibuf));
		HK_TEST_ASSERT(0x70aae483, v.store<3>(ibuf));
		HK_TEST_ASSERT(0x70aae483, v.store<2>(ibuf));
#if !defined(HK_COMPILER_SNC)
		HK_TEST_ASSERT(0x70aae483, v.store<1>(ibuf));
#endif
#endif

		for (int i=0; i<34; ++i) buf[i] = 99;
		ibuf = (hkUint32*)(buf+4);
#if !defined(HK_ALIGN_RELAX_CHECKS)
		HK_TEST_ASSERT(0x70aae483, v.store<4>(ibuf));
		HK_TEST_ASSERT(0x70aae483, v.store<3>(ibuf));
		HK_TEST_ASSERT(0x70aae483, v.store<2>(ibuf));
#endif
		v.store<1>(ibuf);
		HK_TEST(buf[2] == 99);
		HK_TEST(buf[3] == 99);
		HK_TEST(ibuf[0] == 1);
		HK_TEST(buf[8] == 99);
		HK_TEST(buf[9] == 99);

		for (int i=0; i<34; ++i) buf[i] = 99;
		ibuf = (hkUint32*)(buf+8);
#if !defined(HK_ALIGN_RELAX_CHECKS)
#if (HK_NATIVE_ALIGN_CHECK == 0xF)  // Loading from 8-byte alignment
		HK_TEST_ASSERT(0x70aae483, v.store<4>(ibuf));
#endif
		HK_TEST_ASSERT(0x70aae483, v.store<3>(ibuf));
#endif
		v.store<2>(ibuf);
		HK_TEST(buf[6] == 99);
		HK_TEST(buf[7] == 99);
		HK_TEST(ibuf[0] == 1);
		HK_TEST(ibuf[1] == 2);
		HK_TEST(buf[16] == 99);
		HK_TEST(buf[17] == 99);

		for (int i=0; i<34; ++i) buf[i] = 99;
		ibuf = (hkUint32*)(buf+16);
		v.store<3>(ibuf);
		HK_TEST(buf[14] == 99);
		HK_TEST(buf[15] == 99);
		HK_TEST(ibuf[0] == 1);
		HK_TEST(ibuf[1] == 2);
		HK_TEST(ibuf[2] == 3);
		HK_TEST(buf[28] == 99);
		HK_TEST(buf[29] == 99);

		for (int i=0; i<34; ++i) buf[i] = 99;
		ibuf = (hkUint32*)(buf+16);
		v.store<4>(ibuf);
		HK_TEST(buf[14] == 99);
		HK_TEST(buf[15] == 99);
		HK_TEST(ibuf[0] == 1);
		HK_TEST(ibuf[1] == 2);
		HK_TEST(ibuf[2] == 3);
		HK_TEST(ibuf[3] == 4);
		HK_TEST(buf[32] == 99);
		HK_TEST(buf[33] == 99);
	}

	{
		hkIntVector v;
		v.set(1, 2, 3, 4);
		hkUint32* ibuf = (hkUint32*)(buf+4);

		for (int i=0; i<34; ++i) buf[i] = 99;
		v.storeNotAligned<1>(ibuf);
		HK_TEST(buf[2] == 99);
		HK_TEST(buf[3] == 99);
		HK_TEST(ibuf[0] == 1);
		HK_TEST(buf[8] == 99);
		HK_TEST(buf[9] == 99);

		for (int i=0; i<34; ++i) buf[i] = 99;
		v.storeNotAligned<2>(ibuf);
		HK_TEST(buf[2] == 99);
		HK_TEST(buf[3] == 99);
		HK_TEST(ibuf[0] == 1);
		HK_TEST(ibuf[1] == 2);
		HK_TEST(buf[12] == 99);
		HK_TEST(buf[13] == 99);

		for (int i=0; i<34; ++i) buf[i] = 99;
		v.storeNotAligned<3>(ibuf);
		HK_TEST(buf[2] == 99);
		HK_TEST(buf[3] == 99);
		HK_TEST(ibuf[0] == 1);
		HK_TEST(ibuf[1] == 2);
		HK_TEST(ibuf[2] == 3);
		HK_TEST(buf[16] == 99);
		HK_TEST(buf[17] == 99);

		for (int i=0; i<34; ++i) buf[i] = 99;
		v.storeNotAligned<4>(ibuf);
		HK_TEST(buf[2] == 99);
		HK_TEST(buf[3] == 99);
		HK_TEST(ibuf[0] == 1);
		HK_TEST(ibuf[1] == 2);
		HK_TEST(ibuf[2] == 3);
		HK_TEST(ibuf[3] == 4);
		HK_TEST(buf[20] == 99);
		HK_TEST(buf[21] == 99);
	}
}

static void compare()
{
	{
		HK_ALIGN16( hkInt32 a[4]) = {1,1,1,1};
		HK_ALIGN16( hkInt32 b[4]) = {-1,-1,-1,-1};
		HK_ALIGN16( hkInt32 c[4]) = {0,0,0,0};
		HK_TEST( !((hkIntVector*)a)->isNegativeAssumingAllValuesEqual());
		HK_TEST( ((hkIntVector*)b)->isNegativeAssumingAllValuesEqual() );
		HK_TEST( !((hkIntVector*)c)->isNegativeAssumingAllValuesEqual());
	}
	{
		hkIntVector vA;	vA.set(1, 2, 3, 4);
		hkIntVector vB;	vB.set(5, 6, 3, 3);
		hkVector4Comparison cmpLess = vA.compareLessThanS32(vB);
		hkVector4Comparison cmpEq = vA.compareEqualS32(vB);

		HK_TEST(cmpLess.allAreSet<hkVector4ComparisonMask::MASK_XY>());
		HK_TEST(cmpEq.allAreSet<hkVector4ComparisonMask::MASK_Z>());
	}
	{
		hkIntVector vA;	vA.set(1, 2, 3, 4);
		hkIntVector vB;	vB.set(-5, -6, -3, -3);
		hkVector4Comparison cmpLess = vA.compareLessThanS32(vB);
		hkVector4Comparison cmpEq = vA.compareEqualS32(vB);

		HK_TEST(!cmpLess.allAreSet());
		HK_TEST(!cmpEq.allAreSet());
	}
	{
		hkIntVector vA;	vA.set( 0,  1, HK_INT32_MAX, 0xffffffff);
		hkIntVector vB;	vB.set(-0, -1, HK_INT32_MIN, 0x00000000);
		hkVector4Comparison cmpEq = vA.compareEqualS32(vB);

		HK_TEST(cmpEq.allAreSet<hkVector4ComparisonMask::MASK_X>());
	}
}

static void convertF2I()
{
	// should clip at 0 and round towards 0
	{
		hkVector4 f; f.set(1.4f, -13.8f, 70000.0f, -HK_REAL_MAX);
		hkIntVector i; i.setConvertF32toU32(f);
		HK_TEST(i.getU32<0>() == 1);
		HK_TEST(i.getU32<1>() == 0);
		HK_TEST(i.getU32<2>() == 70000);
		HK_TEST(i.getU32<3>() == 0);
	}
	{
		hkVector4 f; f.set(1.8f, -13.4f, -70000.0f, HK_REAL_MAX);
		hkIntVector i; i.setConvertF32toU32(f);
		HK_TEST(i.getU32<0>() == 1);
		HK_TEST(i.getU32<1>() == 0);
		HK_TEST(i.getU32<2>() == 0);
		HK_TEST(i.getU32<3>() >= 0xffffee00); // max normalized mantissa
	}
#ifndef HK_REAL_IS_DOUBLE
	{
		HK_ALIGN_REAL(hkUint32 f[4]) = { 0x00000000, 0x80000000, 0xffffffff, 0x00800000 }; // 0, -0, NaN, smallest normalized pos number
		hkIntVector i; i.setConvertF32toU32(*(hkVector4*)f);
		HK_TEST(i.getU32<0>() == 0);
		HK_TEST(i.getU32<1>() == 0);
		HK_TEST(i.getU32<2>() == 0);
		HK_TEST(i.getU32<3>() == 0);
	}
#endif
	// should round towards 0
	{
		hkVector4 f; f.set(1.4f, -13.8f, 70000.0f, -HK_REAL_MAX);
		hkIntVector i; i.setConvertF32toS32(f);
		hkInt32* ii = (hkInt32*)&i.m_quad;
		HK_TEST(ii[0] == 1);
		HK_TEST(ii[1] == -13);
		HK_TEST(ii[2] == 70000);
		HK_TEST(ii[3] <= int(0x80001200)); // min signed normalized mantissa
	}
	{
		hkVector4 f; f.set(1.8f, -13.4f, -70000.0f, HK_REAL_MAX);
		hkIntVector i; i.setConvertF32toS32(f);
		hkInt32* ii = (hkInt32*)&i.m_quad;
		HK_TEST(ii[0] == 1);
		HK_TEST(ii[1] == -13);
		HK_TEST(ii[2] == -70000);

// Commented out for platforms where casting a large float (>= 2^32) to int produces a different result
#if !defined(HK_PLATFORM_CTR) && !defined(HK_PLATFORM_RVL) && !defined(HK_PLATFORM_ANDROID) && !defined(HK_PLATFORM_TIZEN) && !defined(HK_PLATFORM_LINUX) && defined(HK_REAL_IS_FLOAT)
		HK_TEST(ii[3] >= 0x7fffee00); // max signed normalized mantissa
#endif
	}
#ifndef HK_REAL_IS_DOUBLE
	{
		HK_ALIGN_REAL(hkUint32 f[4]) = { 0x00000000, 0x80000000, 0xffffffff, 0x00800000 }; // 0, -0, NaN, smallest normalized pos number
		hkIntVector i; i.setConvertF32toS32(*(hkVector4*)f);
		hkInt32* ii = (hkInt32*)&i.m_quad;
		HK_TEST(ii[0] == 0);
		HK_TEST(ii[1] == 0);
		// ii[2] undefined behavior: win32 == HK_INT32_MIN  360 == 0x0
		HK_TEST(ii[3] == 0);
	}
#endif
}

//
//	Assuming that this = (i0, i1, i2, i3) and v = (v0, v1, v2, v3), the function will return ik
//	where k in {0,..., 3} such that vk = max{v0, v1, v2, v3}.
// In case of equality, returns last index (same as hkVector4)

static int hkIntVector_getComponentAtVectorMax(const hkIntVector& vec, hkVector4Parameter v)
{
	hkQuadIntUnion qu;
	qu.q = vec.m_quad;

	const hkReal vx = v(0);
	const hkReal vy = v(1);

	hkReal maxXY;
	int idxXY;
	if ( vy >= vx )	{	maxXY = vy;		idxXY = qu.u[1];	}
	else			{	maxXY = vx;		idxXY = qu.u[0];	}

	const hkReal vz = v(2);
	const hkReal vw = v(3);

	hkReal maxZW;
	int idxZW;
	if ( vw >= vz )	{	maxZW = vw;		idxZW = qu.u[3];	}
	else			{	maxZW = vz;		idxZW = qu.u[2];	}

	return ( maxZW >= maxXY ) ? idxZW : idxXY;
}

//
//	Misc ops

static void miscTests()
{
	// Set
	{
		hkIntVector v;
		v.setAll(99);
		HK_TEST(v.getU32<0>() == 99);
		HK_TEST(v.getU32<1>() == 99);
		HK_TEST(v.getU32<2>() == 99);
		HK_TEST(v.getU32<3>() == 99);
	}
	{
		HK_ALIGN16( hkUint32 v[4]) = { 1,2,3,4 };
		((hkIntVector*)v)->setFirstComponent(99);
		HK_TEST(v[0] == 99);
		HK_TEST(v[1] ==  0);
		HK_TEST(v[2] ==  0);
		HK_TEST(v[3] ==  0);
	}
	{
		hkIntVector v;
		v.set(1, 2, 3, 4);
		HK_TEST(v.getU32<0>() == 1);
		HK_TEST(v.getU32<1>() == 2);
		HK_TEST(v.getU32<2>() == 3);
		HK_TEST(v.getU32<3>() == 4);
		int a = 0;
		HK_TEST_ASSERT(0xfabb2300, a = v.getU32(5));
		v.setAll(a);
	}

	// Get constant
	{
		hkIntVector v = hkIntVector::getConstant<HK_QUADINT_0123>();
		HK_TEST(v.getU32<0>() == 0);
		HK_TEST(v.getU32<1>() == 1);
		HK_TEST(v.getU32<2>() == 2);
		HK_TEST(v.getU32<3>() == 3);
		hkBool32 neg = 0;
		HK_TEST_ASSERT(0x252d00fa, neg = v.isNegativeAssumingAllValuesEqual());
		v.setAll((int)neg);
	}

	// Broadcast max component
	{
		hkVector4 vecs[11];
		hkIntVector idxs[11];

		vecs[ 0].set(1.0f, 2.0f, 3.0f, 4.0f);		idxs[ 0].setAll(3);
		vecs[ 1].set(2.0f, 3.0f, 4.0f, 1.0f);		idxs[ 1].setAll(2);
		vecs[ 2].set(3.0f, 4.0f, 1.0f, 2.0f);		idxs[ 2].setAll(1);
		vecs[ 3].set(4.0f, 1.0f, 2.0f, 3.0f);		idxs[ 3].setAll(0);
		vecs[ 4].set(4.0f, 2.0f, 3.0f, 4.0f);		idxs[ 4].setAll(3);
		vecs[ 5].set(1.0f, 4.0f, 3.0f, 4.0f);		idxs[ 5].setAll(3);
		vecs[ 6].set(1.0f, 2.0f, 4.0f, 4.0f);		idxs[ 6].setAll(3);
		vecs[ 7].set(4.0f, 4.0f, 4.0f, 3.0f);		idxs[ 7].setAll(2);
		vecs[ 8].set(4.0f, 2.0f, 4.0f, 2.0f);		idxs[ 8].setAll(2);
		vecs[ 9].set(1.0f, 4.0f, 1.0f, 1.0f);		idxs[ 9].setAll(1);
		vecs[10].set(4.0f, 4.0f, 4.0f, 4.0f);		idxs[10].setAll(3);

		for (int i = 0; i < 11; i++)
		{
			hkVector4 v = vecs[i];
			hkIntVector ii;
			ii.set(0, 1, 2, 3);

			ii.broadcastComponentAtVectorMax(v);
			hkVector4Comparison cmp = ii.compareEqualS32(idxs[i]);
			HK_TEST(cmp.allAreSet());

			ii.set(0, 1, 2, 3);
			int kk = ii.getComponentAtVectorMax(v);
			int kkk = hkIntVector_getComponentAtVectorMax(ii, v);
			HK_TEST(kk == kkk);
			HK_TEST(idxs[i].getU32<0>() == (hkUint32)kk);
		}
	}

	// Set select
	{
		hkIntVector vA;	vA.set(1, 2, 3, 4);
		hkIntVector vB;	vB.set(4, 3, 2, 1);

		hkVector4Comparison cmp;
		cmp.set<hkVector4ComparisonMask::MASK_XY>();
		hkIntVector vC;	vC.setZero(); vC.setSelect(cmp, vA, vB);
		hkIntVector vD;	vD.set(1, 2, 2, 1);

		cmp = vC.compareEqualS32(vD);
		HK_TEST(cmp.allAreSet());
	}

	// Convert u16 to u32
	{
		hkIntVector v;

		hkUint16 data[] = {
			(hkUint16)0xF001, (hkUint16)0xF002, (hkUint16)0xF003, (hkUint16)0xF004, 
			(hkUint16)0xF005, (hkUint16)0xF006, (hkUint16)0xF007, (hkUint16)0xF008
		};

		v.load<4, HK_IO_NATIVE_ALIGNED>(reinterpret_cast<hkUint32*>(data));

		union{
			hkIntVector vec;
			hkUint32 vals[4];
		} vLo, vHi;
 
		vLo.vec.setConvertLowerU16ToU32(v);
		vHi.vec.setConvertUpperU16ToU32(v);

		bool blo = (vLo.vals[0] == 0xF001) && (vLo.vals[1] == 0xF002) && (vLo.vals[2] == 0xF003) && (vLo.vals[3] == 0xF004);
		bool bhi = (vHi.vals[0] == 0xF005) && (vHi.vals[1] == 0xF006) && (vHi.vals[2] == 0xF007) && (vHi.vals[3] == 0xF008);

		HK_TEST(blo);
		HK_TEST(bhi);
	}
}

template< int N >
static void loadByteAligned()
{
	HK_ALIGN16( char bytes[32] );
	for (int i=0; i<32; i++)
	{
		bytes[i] = (char) i+1;
	}

	for (int i=0; i<16; i++)
	{
		hkIntVector loadedByteAligned, memCopied;
		loadedByteAligned.setZero();
		memCopied.setZero();
		loadedByteAligned.load<N, HK_IO_BYTE_ALIGNED>( (hkUint32*)(bytes+i) );
		hkString::memCpy( &memCopied, bytes+i, sizeof(hkIntVector) );
		
		if(N >= 1) HK_TEST2( memCopied.getComponent<0>() == loadedByteAligned.getComponent<0>(), "loadNotAligned failed for N=" << N );
		if(N >= 2) HK_TEST2( memCopied.getComponent<1>() == loadedByteAligned.getComponent<1>(), "loadNotAligned failed for N=" << N );
		if(N >= 3) HK_TEST2( memCopied.getComponent<2>() == loadedByteAligned.getComponent<2>(), "loadNotAligned failed for N=" << N );
		if(N >= 4) HK_TEST2( memCopied.getComponent<3>() == loadedByteAligned.getComponent<3>(), "loadNotAligned failed for N=" << N );
	}
}

template< int N >
static void loadWordAligned()
{
	HK_ALIGN16( hkUint32 words[8] );
	for (int i=0; i<8; i++)
	{
		words[i] = (hkUint32) i+1;
	}

	for (int i=0; i<4; i++)
	{
		hkIntVector loadedWordAligned, memCopied;
		loadedWordAligned.setZero();
		memCopied.setZero();
		loadedWordAligned.load<N, HK_IO_NATIVE_ALIGNED>( words + i );
		hkString::memCpy( &memCopied, words+i, sizeof(hkIntVector) );

		if(N >= 1) HK_TEST2( memCopied.getComponent<0>() == loadedWordAligned.getComponent<0>(), "loadWordAligned failed for N=" << N );
		if(N >= 2) HK_TEST2( memCopied.getComponent<1>() == loadedWordAligned.getComponent<1>(), "loadWordAligned failed for N=" << N );
		if(N >= 3) HK_TEST2( memCopied.getComponent<2>() == loadedWordAligned.getComponent<2>(), "loadWordAligned failed for N=" << N );
		if(N >= 4) HK_TEST2( memCopied.getComponent<3>() == loadedWordAligned.getComponent<3>(), "loadWordAligned failed for N=" << N );
	}
}

template< int N >
static void storeByteAligned()
{
	const int bufferSize = 32;
	HK_ALIGN16( char memCpyDest[bufferSize] );
	HK_ALIGN16( char storeDest[bufferSize] );

	hkIntVector testVec = hkIntVector::getConstant<HK_QUADINT_0123>();

	for (int j=0; j<16; j++)
	{
		// Reset the buffers
		for (int i=0; i<bufferSize; i++)
		{
			memCpyDest[i] = (char) i+127;
			storeDest[i] = (char) i+127;
		}

		// Write with the int vector
		testVec.store<N, HK_IO_BYTE_ALIGNED>( (hkUint32*) (storeDest+j) );
		// Write with memcpy
		hkString::memCpy( memCpyDest+j, &testVec, N * sizeof(hkUint32) );

		// Make sure the results are equal. This will check that the right values got written, 
		// and nothing out of range got touched.
		for (int i=0; i<bufferSize; i++)
		{
			HK_TEST2( storeDest[i] == memCpyDest[i], "storeNotAligned failed for N=" << N ); 
			if (storeDest[i] != memCpyDest[i])
				break;
		}
	}
}

template< int N >
static void storeWordAligned()
{
	const int bufferSize = 8;
	HK_ALIGN16( hkUint32 memCpyDest[bufferSize] );
	HK_ALIGN16( hkUint32 storeDest[bufferSize] );

	hkIntVector testVec = hkIntVector::getConstant<HK_QUADINT_0123>();

	for (int j=0; j<4; j++)
	{
		// Reset the buffers
		for (int i=0; i<bufferSize; i++)
		{
			memCpyDest[i] = (char) i+127;
			storeDest[i] = (char) i+127;
		}

		// Write with the int vector
		testVec.store<N, HK_IO_NATIVE_ALIGNED>( storeDest+j );
		// Write with memcpy
		hkString::memCpy( memCpyDest+j, &testVec, N * sizeof(hkUint32) );

		// Make sure the results are equal. This will check that the right values got written, 
		// and nothing out of range got touched.
		for (int i=0; i<bufferSize; i++)
		{
			HK_TEST2( storeDest[i] == memCpyDest[i], "storeNotAligned failed for N=" << N ); 
			if (storeDest[i] != memCpyDest[i])
				break;
		}
	}
}

template< int N >
static void testLoadStore()
{
	loadByteAligned<N>();
	loadWordAligned<N>();
	storeByteAligned<N>();
	storeWordAligned<N>();
}

int VectorUint4_main()
{
	{ 
		testLoadStore<1>();
		testLoadStore<2>();
		testLoadStore<3>();
		testLoadStore<4>();

		loadAndSet();
		store();
		binaryOperations();
		arithmetic();
		shift();
		merge_pack();
		select();
		shuffle();
		permute();
		compare();
		convertF2I();
		convertI2F();
		miscTests();
	}
	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#	pragma force_active on
#endif
HK_TEST_REGISTER(VectorUint4_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
