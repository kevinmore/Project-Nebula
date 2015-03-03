/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Vector/hkFourTransposedPointsd.h>
#include <Geometry/Internal/Types/hkcdVertex.h>
#include <Common/Base/Math/Vector/hkIntVector.h>
//
//	Sets this = Transpose(m) * v

void hkFourTransposedPointsd::setTransformedInverseDir(const hkMatrix3d& m, const hkFourTransposedPointsd& v)
{
	_setTransformedInverseDir(m, v);
}

//
//	Sets this = Transpose(Rotation(m)) * (v - Translation(m))

void hkFourTransposedPointsd::setTransformedInversePos(const hkTransformd& m, const hkFourTransposedPointsd& v)
{
	_setTransformedInversePos(m, v);
}

//
//	Sets this = m * v

void hkFourTransposedPointsd::setRotatedDir(const hkMatrix3d& m, const hkFourTransposedPointsd& v)
{
	_setRotatedDir(m, v);
}

//
//	Copies the points from a hkVector4d array into a hkFourTransposedPointsd array

void HK_CALL hkFourTransposedPointsd::copyVertexData(const hkVector4d* HK_RESTRICT src, const int numVerts, hkArray<hkFourTransposedPointsd>& pointsOut)
{
	const int paddedSize = HK_NEXT_MULTIPLE_OF(4, numVerts);
	const int numBatches = numVerts >> 2;

	hkFourTransposedPointsd* HK_RESTRICT dst = pointsOut.expandBy(paddedSize >> 2);
	for (int bi = 0; bi < numBatches; bi++, src += 4)
	{
		dst[bi].set(src[0], src[1], src[2], src[3]);
	}

	// Last batch
	const int numRemaining = numVerts - (numBatches << 2);
	if ( numRemaining )
	{
		// Get remaining vertices
		hkVector4d verts[4];
		for (int i = 0; i < numRemaining; i++)
		{
			verts[i] = src[i];
		}

		// Fill rest with the last vertex
		for (int i = numRemaining; i < 4; i++)
		{
			verts[i] = src[numRemaining - 1];
		}

		dst[numBatches].set(verts[0], verts[1], verts[2], verts[3]);
	}
}

#	if defined(HK_REAL_IS_DOUBLE)
void hkFourTransposedPointsd::getOriginalVertices( const hkFourTransposedPointsd* verts4in, int numVertices, hkcdVertex* verticesOut )
{
	const hkFourTransposedPointsd* HK_RESTRICT vIn = verts4in;
	hkcdVertex* HK_RESTRICT vOut = verticesOut;
	int numFours = int( unsigned(numVertices+3) / 4 );
	hkIntVector indices = hkIntVector::getConstant(HK_QUADINT_0123);
	hkIntVector indicesStep; indicesStep.splatImmediate32<4>();
	for( int bi = 0; bi < numFours; bi++ )
	{
		hkVector4d vi; indices.storeInto24LowerBitsOfReal(vi);

		vIn[bi].extractWithW( vi, vOut[0], vOut[1], vOut[2], vOut[3]);
		vOut += 4;
		indices.setAddU32(indices, indicesStep);
	}
	int nv = numVertices; // ensure the padding vertices duplicate the last one
	for( int i = nv; i < numFours*4; ++i )
	{
		verticesOut[i] = verticesOut[nv-1];
	}
}
#endif

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
