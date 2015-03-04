/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CONVEX_SHAPE_INL
#define HKNP_CONVEX_SHAPE_INL

#include <Common/Base/Math/Vector/Mx/hkMxVector.h>
#include <Common/Base/Math/Vector/Mx/hkMxVectorUtil.h>
#include <Common/Base/Math/Vector/Mx/hkMxUnroll.h>


namespace
{
	// Try to optimized the volume spanned by two edges and a normal.
	// Simply cross the edge vectors and dot with the normal to find the volume.
	template<int typei>
	static HK_FORCE_INLINE void findBestVolume(hkUint8 i,
		hkVector4Parameter normal, hkVector4Parameter triNormal,
		const hkcdVertex* vertices, const hkUint8* indices, const hkVector4* edges,
		hkSimdRealParameter extraOffset, hkSimdRealParameter extraVolume, 
		hkSimdReal& maxVol, int& maxType, hkUint8& maxIndex)
	{
		hkVector4 edge; edge.setSub(vertices[i], vertices[indices[typei]]);
		edge.addMul(normal, extraOffset); // Offset a little bit in the direction of the normal

		hkVector4 area; area.setCross(edges[typei], edge);
		area.normalizeIfNotZero<3>();
		hkSimdReal vol = area.dot<3>(triNormal) + extraVolume; // Add a little bit of extra volume
		if(vol > maxVol)
		{
			maxVol = vol;
			maxIndex = i;
			maxType = typei + 1;
		}
	}

	// Compute the supporting face using vertex information only.
	// The returned face id is a concatenation of the four vertex id's.
	static HK_FORCE_INLINE hkUint32 getSupportingFaceId(
		const hkcdGsk::Cache::VertexId* vertexIds, hkVector4Parameter point, hkVector4Parameter normal,
		const hkcdVertex* vertices, int numVertices, int dim )
	{
		HK_ASSERT2(0x34985AAF, numVertices >= 3, "Too few vertices");
		hkUint8 indices[] = { vertexIds[0], vertexIds[1], vertexIds[2], vertexIds[2] }; 

		hkSimdReal extraOffset = hkSimdReal::fromFloat(0.01f); 

		hkUint32 faceIdResult = 0;

		switch(dim)
		{
		case 1:
			{
				// Find the edge that is most perpendicular to the normal:
				// -Find the one with smallest squared dot product with the normal
				hkSimdReal bestCosAngleSquared = hkSimdReal_Max;
				hkSimdReal bestSqrLength = hkSimdReal_1;

				hkUint8 maxIx = hkUint8(~0);
				hkVector4 edge0;
				for (hkUint8 i = 0; i < numVertices; i++)
				{
					if ( i == indices[0] )
					{
						continue;
					}

					// Find edge vector
					edge0.setSub(vertices[i], vertices[indices[0]]);
					edge0.addMul(normal, extraOffset); // Offset a little bit in the direction of the normal

					// Find the smallest dot product with normal, but we can just take dot squared instead of normalizing
					// dot_normalized * dot_normalized = dotSquared / sqrLength;
					hkSimdReal cosAngle = normal.dot<3>(edge0);
					hkSimdReal cosAngleSquared = cosAngle * cosAngle;
					cosAngleSquared.setFlipSign(cosAngleSquared, cosAngle);
					hkSimdReal sqrLength = edge0.lengthSquared<3>();

					// Instead of checking two ratios, we can check two products
					// ((a / b) < (c / d)) == ((a * d) < (c * b))
					if (cosAngleSquared * bestSqrLength < bestCosAngleSquared * sqrLength)
					{
						bestCosAngleSquared = cosAngleSquared;
						bestSqrLength = sqrLength;
						maxIx = i;
					}
				}

				indices[1] = maxIx;
				indices[2] = maxIx;
				// Note: No break!
			}

		case 2:
			{
				// For a triangle with the two previously found points and a new point
				// Find the vertex that creates the largest volume when using the triangle as base and the normal as height
				// The edge also needs to as close to perpendicular to the normal as possible
				// - Find the edge with largest dot product with the cross product of the normal and the first edge
				// - The edge also needs to have the smallest dot product with the normal
				//
				
				
				hkSimdReal bestAbsVol = -hkSimdReal_Max;
				hkSimdReal bestAbsCosAngle = hkSimdReal_1;

				hkVector4 edge0; edge0.setSub(vertices[indices[1]], vertices[indices[0]]);
				HK_ASSERT2(0x34985AB0, edge0.lengthSquared<3>().isGreater(hkSimdReal_Eps), "Degenerated line");
				edge0.normalize<3, HK_ACC_12_BIT, HK_SQRT_IGNORE>();

				// Rotate the normal to align the edge perpendicularly
				hkVector4 edgeNormal; edgeNormal.setSubMul(normal, edge0, normal.dot<3>(edge0));
				HK_ASSERT2(0x34985AB0, edgeNormal.lengthSquared<3>().isGreater(hkSimdReal_Eps), "Degenerated normal");
				edgeNormal.normalize<3, HK_ACC_12_BIT, HK_SQRT_IGNORE>();

				// Note: a . (b x c) = - c . (b x a)
				// but we will take absolute volume, so we can cross the normal with an edge first
				hkVector4 area; area.setCross(edgeNormal, edge0);

				hkUint8 minIx = hkUint8(~0);
				hkVector4 edge1;
				for(hkUint8 i=0; i<numVertices; i++)
				{
					if (( i == indices[1] ) || ( i == indices[0] ))
					{
						continue;
					}

					edge1.setSub(vertices[i], vertices[indices[0]]);
					edge1.addMul(normal, extraOffset); // Offset a little bit in the direction of the normal

					hkSimdReal vol = area.dot<3>(edge1);
					hkSimdReal cosAngle = edgeNormal.dot<3>(edge1);

					hkSimdReal absVol; absVol.setAbs(vol);
					hkSimdReal absCosAngle; absCosAngle.setAbs(cosAngle);

					// Instead of checking two ratios, we can check two products
					// ((a / b) > (c / d)) == ((a * d) > (c * b))
					if (absVol * bestAbsCosAngle > bestAbsVol * absCosAngle)
					{
						bestAbsVol = absVol;
						bestAbsCosAngle = absCosAngle;
						minIx = i;
					}
				}

				indices[2] = minIx;
				// Note: No break!
			}

		case 3:
			{
				// For a triangle with two of the previously found points, (3 choose 2) = 3 types, and a new point
				// Find the vertex that creates the largest volume when using the triangle as base and the normal as height
				int maxType = 0;
				hkSimdReal maxVol = -hkSimdReal_Max;

				// Compute the triangle in clockwise orientation
				hkVector4 edges[3];
				hkVector4 triNormal;
				{
					edges[0].setSub(vertices[indices[1]], vertices[indices[0]]);
					edges[1].setSub(vertices[indices[2]], vertices[indices[1]]);
					edges[2].setSub(vertices[indices[0]], vertices[indices[2]]);

					triNormal.setCross(edges[1], edges[0]);
					triNormal.normalizeIfNotZero<3, HK_ACC_12_BIT, HK_SQRT_IGNORE>();

					// Fix triangle winding (note: a . (b x c) = - c . (b x a) )
					hkSimdReal s = triNormal.dot<3>(normal);
					bool flip = s.isGreaterEqualZero();
					if (flip)
					{
						hkAlgorithm::swap(indices[1], indices[2]);
						hkAlgorithm::swap(edges[0], edges[2]);

						hkVAR_UNROLL(3, edges[hkI].setNeg<3>(edges[hkI]));

						triNormal.setNeg<3>(triNormal);
					}

					hkVAR_UNROLL(3, edges[hkI].normalizeIfNotZero<3>());
				}

				hkUint8 maxIx = hkUint8(~0);
				for (hkUint8 i = 0; i < numVertices; i++)
				{
					if (( i == indices[2] ) || ( i == indices[1] ) || ( i == indices[0] ))
					{
						continue;
					}

					hkSimdReal extraVolume = hkSimdReal::fromFloat(0.0001f * i); 
					hkVAR_UNROLL(3, findBestVolume<hkI>(i, normal, triNormal, vertices, indices, edges, extraOffset, extraVolume, maxVol, maxType, maxIx));
				}

				// Make a triangle unless we can find a quad
				indices[3] = indices[2];

				bool ok = maxVol.getReal() > 0.9f; 
				if (ok)
				{
					// Insert vertex at the right position
					// ..first move other vertices to make room
					for (int j = 3; j > maxType; j--)
					{
						indices[j] = indices[j-1];
					}
					indices[maxType] = maxIx;
				}

				// Make sure we always get them in the same order (rotate positions so that first vertex has lowest index)
				int minVertexIx = 1000;
				int minVertexIxIx = -1;
				for (int j = 0; j < 4; j++)
				{
					if (indices[j] < minVertexIx)
					{
						minVertexIx = indices[j];
						minVertexIxIx = j;
					}
				}

				hkUint8 newVertexIndices[4];
				for (int j = 0; j < 4; j++)
				{
					newVertexIndices[j] = indices[(j + minVertexIxIx) & 0x3];
				}
				for (int j = 0; j < 4; j++)
				{
					indices[j] = newVertexIndices[3-j];
				}

				// Temp fix for invalid vertex ids
				for (int j = 0; j < 4; j++)
				{
					if(indices[j] == hkUint8(~0))
					{
						for (int d = 1; d < 4; d++)
						{
							hkUint8 ix = indices[(j + d) & 0x3];
							if(ix != hkUint8(~0))
							{
								indices[j] = ix;
								break;
							}
						}
					}
				}

				faceIdResult = (indices[0] << 24) | (indices[1] << 16) | (indices[2] << 8) | (indices[3]);
			}
		}

		return faceIdResult;
	}

	//
	static HK_FORCE_INLINE hkUint32 revertToPrevFaceIdIfNeeded(
		const hknpConvexShape* HK_RESTRICT shape,
		hkUint32 faceIdB, hkVector4Parameter planeBinA, hkVector4Parameter gskPoint_local, hkUint32 &origFaceIdB )
	{
		if(origFaceIdB != 0xFFFFFFFF)
		{
			HK_ASSERT2(0xf077e3da, ((faceIdB >> 24) & 0xff)		< (hkUint32)shape->getNumberOfVertices(),	"Invalid vertex index");
			HK_ASSERT2(0xf077e3da, ((faceIdB >> 16) & 0xff)		< (hkUint32)shape->getNumberOfVertices(),	"Invalid vertex index");
			HK_ASSERT2(0xf077e3da, ((faceIdB >> 8)  & 0xff)		< (hkUint32)shape->getNumberOfVertices(),	"Invalid vertex index");
			HK_ASSERT2(0xf077e3da, ((faceIdB)       & 0xff)		< (hkUint32)shape->getNumberOfVertices(),	"Invalid vertex index");
			HK_ASSERT2(0xf077e3da, ((origFaceIdB >> 24) & 0xff) < (hkUint32)shape->getNumberOfVertices(),	"Invalid vertex index");
			HK_ASSERT2(0xf077e3da, ((origFaceIdB >> 16) & 0xff) < (hkUint32)shape->getNumberOfVertices(),	"Invalid vertex index");
			HK_ASSERT2(0xf077e3da, ((origFaceIdB >> 8)  & 0xff) < (hkUint32)shape->getNumberOfVertices(),	"Invalid vertex index");
			HK_ASSERT2(0xf077e3da, ((origFaceIdB)       & 0xff) < (hkUint32)shape->getNumberOfVertices(),	"Invalid vertex index");

			int dummy;
			hkVector4 prevPlane;
			shape->getFaceInfo(origFaceIdB, prevPlane, dummy);
			hkSimdReal dot = prevPlane.dot<3>(planeBinA);

			if(dot.getReal() > 0.99f)
			{
				hkVector4 plane2;
				hkcdVertex verticesB_inB[4];
				shape->getFaceVertices( origFaceIdB, plane2, verticesB_inB );
				hkVector4 gskPoint_inB = gskPoint_local;
				// Now check whether gskPoint_inA is supported by the quad formed by verticesB_inA

				// Check first triangle of the quad
				bool inside1;
				{
					hkVector4 verticesB_inA_0[4];
					hkVector4 gskPoint_inA_0;

					for(int i=0; i<4; i++)
					{
						verticesB_inA_0[i].setSub(verticesB_inB[i], verticesB_inB[0]);
					}
					gskPoint_inA_0.setSub(gskPoint_inB, verticesB_inB[0]);

					hkMatrix3 M;
					hkVector4 n; n.setCross(verticesB_inA_0[1], verticesB_inA_0[3]);
					M.setCols(verticesB_inA_0[1], verticesB_inA_0[3], n);
					M.invert(1e-6f);
					hkVector4 result;
					M.multiplyVector(gskPoint_inA_0, result);
					inside1 =
						result.getComponent<0>().getReal() >= -1e-3f
						&& result.getComponent<1>().getReal() >= -1e-3f
						&& result.getComponent<0>().getReal() + result.getComponent<1>().getReal() <= 1.001f;
				}
				if(inside1)
				{
					faceIdB = origFaceIdB;
					return faceIdB;
				}

				// Check second triangle of the quad
				bool inside2;
				{
					hkVector4 verticesB_inA_0[4];
					hkVector4 gskPoint_inA_0;

					for(int i=0; i<4; i++)
					{
						verticesB_inA_0[i].setSub(verticesB_inB[i], verticesB_inB[2]);
					}
					gskPoint_inA_0.setSub(gskPoint_inB, verticesB_inB[2]);

					hkMatrix3 M;
					hkVector4 n; n.setCross(verticesB_inA_0[1], verticesB_inA_0[3]);
					M.setCols(verticesB_inA_0[1], verticesB_inA_0[3], n);
					M.invert(1e-6f);
					hkVector4 result;
					M.multiplyVector(gskPoint_inA_0, result);
					inside2 =
						result.getComponent<0>().getReal() >= -1e-3f
						&& result.getComponent<1>().getReal() >= -1e-3f
						&& result.getComponent<0>().getReal() + result.getComponent<1>().getReal() <= 1.001f;
				}

				if(inside2)
				{
					faceIdB = origFaceIdB;
					return faceIdB;
				}
			}
		}
		return faceIdB;
	}

}	// anonymous namespace


#if !defined(HK_PLATFORM_SPU)

HK_FORCE_INLINE hknpConvexShape::hknpConvexShape( int numVertices, hkReal radius, int sizeOfBaseClass )
	: hknpShape( hknpCollisionDispatchType::CONVEX )
{
	init( numVertices, radius, sizeOfBaseClass );
}

HK_FORCE_INLINE void* hknpConvexShape::allocateConvexShape( int numVertices, int sizeOfBaseClass, int& shapeSizeOut )
{
	// Calculate required memory foot-print
	shapeSizeOut = calcConvexShapeSize( numVertices, sizeOfBaseClass );
	HK_ASSERT2(0xf034e3de, shapeSizeOut < 32767, "Shape too large to fit in memory");

	// Allocate
	void* block = hkMemoryRouter::getInstance().heap().blockAlloc( shapeSizeOut );
	HK_MEMORY_TRACKER_ON_NEW_REFOBJECT( shapeSizeOut, block );

	return block;
}

#endif	//!HK_PLATFORM_SPU


HK_FORCE_INLINE int hknpConvexShape::calcConvexShapeSize( int numVertices, int sizeofBaseClass )
{
	const int baseSize	= HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeofBaseClass);
	const int vertsSize	= HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, ((numVertices >> 2) * sizeof(hkMatrix4)));
	const int totalSize	= HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, baseSize + vertsSize);
	return totalSize;
}

HK_FORCE_INLINE hknpConvexShape* hknpConvexShape::createInPlace(
	int numVertices, hkReal radius, hkUint8* buffer, int bufferSize, int sizeOfBaseClass )
{
#if !defined(HK_ALIGN_RELAX_CHECKS)
	HK_ASSERT2(0x15249213, !(hkUlong(buffer) & 0xF), "Shape buffer must be 16 byte aligned");
#endif
	HK_ON_DEBUG(const int shapeSize = calcConvexShapeSize(numVertices, sizeOfBaseClass));
	HK_ASSERT2(0x30a392a2, shapeSize <= bufferSize, "Shape too large to fit in buffer");

#if !defined(HK_PLATFORM_SPU)
	// Construct in-place
	hknpConvexShape* convex = new (buffer) hknpConvexShape(numVertices, radius, sizeOfBaseClass);
#else
	// Initialize manually to avoid creating the v-table
	hknpConvexShape* convex = reinterpret_cast<hknpConvexShape*>(buffer);
	convex->hknpShape::init(hknpCollisionDispatchType::CONVEX);
	convex->init(numVertices, radius, sizeOfBaseClass);

	hknpShapeVirtualTableUtil::patchVirtualTableWithType<hknpShapeType::CONVEX>(convex);
#endif

	convex->m_memSizeAndFlags = 0;
	return convex;
}

HK_FORCE_INLINE void hknpConvexShape::init( int numVertices, hkReal radius, int sizeOfBaseClass )
{
	HK_ASSERT(0xF973BE7F, (sizeOfBaseClass - HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeOfBaseClass)) == 0);
	HK_ASSERT2(0xF973BE7C, (numVertices & 3) == 0, "Vertices must be padded");
	HK_ASSERT2(0x9FC800A4, numVertices < 256, "Too many vertices");
	HK_ASSERT2(0x4AB28203, radius >= 0, "Radius must be positive or zero");

	m_flags.orWith(IS_CONVEX_SHAPE);
	m_convexRadius = radius;

	// Compute stream offsets
	hkUint16 verticesOffset	= (hkUint16)HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeOfBaseClass);
	m_vertices._setOffset(verticesOffset - HK_OFFSET_OF(hknpConvexShape, m_vertices));
	m_vertices._setSize((hkUint16) numVertices);
}

HK_FORCE_INLINE void hknpConvexShape::calcAabbNoRadius( const hkTransform& transform, hkAabb& aabbOut ) const
{
	hkAabb aabb; aabb.setEmpty();
	const int MXSIZE = 4;
	const hkVector4* HK_RESTRICT verts = getVertices();
	for( int i=0; i < hknpConvexShape::getNumberOfVertices(); verts += MXSIZE, i+= MXSIZE )
	{
		// transform vertices to requested space
		hkMxVector<MXSIZE> vertsMx; vertsMx.moveLoad( verts );
		hkMxVector<MXSIZE> positions;
		hkMxVectorUtil::rotateDirection( (const hkMxTransform&)transform, vertsMx, positions );
		hkMxUNROLL_4( aabb.includePoint( positions.getVector<hkMxI>() ) );
	}

	hkVector4 offset = transform.getTranslation();
	aabbOut.m_min.setAdd( aabb.m_min, offset );
	aabbOut.m_max.setAdd( aabb.m_max, offset );
}

HK_DISABLE_OPTIMIZATION_VS2008_X64	// this compiles badly on Visual Studio 2008 x64
HK_FORCE_INLINE void hknpConvexShape::getSupportingVertex(
	hkVector4Parameter direction, hkcdVertex* HK_RESTRICT svOut ) const
{
	const hkVector4* HK_RESTRICT fvA = getVertices();
	hk4xVector4 bestVertA;	bestVertA.moveLoad( fvA );
	hk4xSingle directionAVec; directionAVec.setVector(direction);
	hk4xReal bestDotA; bestDotA.setDot<3>( directionAVec, bestVertA  );

	fvA += 4;

	int a = getNumberOfVertices() - 4;

	while( a > 0 )
	{
		hk4xVector4 vertA;	vertA.moveLoad(fvA);
		hk4xReal curDotA; curDotA.setDot<3>( directionAVec, vertA  );
		//hk4xVector4 tttt(curDotA);
		hk4xVector4 cmp; cmp.setBroadcast(curDotA);
		hk4xMask compA; bestDotA.less( cmp, compA );

		bestDotA.setSelect ( compA, curDotA, bestDotA  );
		bestVertA.setSelect( compA, vertA,   bestVertA );

		fvA+=4;
		a -= 4;
	}

	// binary combine a
	hkVector4 bestA;
	{
		hkVector4Comparison cmp1gt0 = bestDotA.getReal<1>().greater( bestDotA.getReal<0>() );
		hkVector4Comparison cmp3gt2 = bestDotA.getReal<3>().greater( bestDotA.getReal<2>() );

		hkSimdReal bestDotA01;  bestDotA01.setSelect( cmp1gt0, bestDotA.getReal<1>(),    bestDotA.getReal<0>() );
		hkVector4 bestVertA01; bestVertA01.setSelect( cmp1gt0, bestVertA.getVector<1>(), bestVertA.getVector<0>() );
		hkSimdReal bestDotA23;  bestDotA23.setSelect( cmp3gt2, bestDotA.getReal<3>(),    bestDotA.getReal<2>() );
		hkVector4 bestVertA23; bestVertA23.setSelect( cmp3gt2, bestVertA.getVector<3>(), bestVertA.getVector<2>() );

		hkVector4Comparison a01GTa23 = bestDotA01.greater(bestDotA23);
		bestA.setSelect( a01GTa23, bestVertA01,  bestVertA23 );
	}
	*(hkVector4*)svOut = bestA;

#if defined(HK_DEBUG)
	if (1)
	{
		hkcdVertex cA; getSupportingVertexRef( direction, cA );
		volatile hkReal vA = direction.dot<3>( cA ).getReal();
		volatile hkReal vB = direction.dot<3>( *svOut ).getReal();
		HK_ASSERT(0x5a7aca40, vA <= HK_REAL_EPSILON + vB || hkMath::abs(vA-vB) <= 1e-6f );
	}
#endif
}
HK_RESTORE_OPTIMIZATION_VS2008_X64

HK_FORCE_INLINE void hknpConvexShape::convertVertexIdsToVertices(
	const hkUint8* ids, int numIds, hkcdVertex* HK_RESTRICT verticesOut ) const
{
	HK_COMPILE_TIME_ASSERT( sizeof(hkVector4) == sizeof(hkcdVertex) );
	HK_ASSERT( 0xf0354767, numIds <= 3 );
	const hkVector4* vertices = getVertices();
#if 0
	do
	{
		const int index = *ids++;
		HK_ASSERT2(0xA17A384A,index<getNumVertices(),"Invalid cache vertex index");
		*verticesOut++ = *(const hkcdVertex*)&vertices[index];
		HK_ASSERT2(0xA17A384B,verticesOut[-1].getInt24W()==index,"Invalid vertex index");
	} while (--numVerts);
#else
	static hkUint8 mask[6] = { 0,0,0, 0xff, 0xff, 0xff };
	hkVector4 a = vertices[ ids[0] /*& mask[ numVerts+2 ]*/ ];
	hkVector4 b = vertices[ ids[1] & mask[ numIds+1 ] ];
	hkVector4 c = vertices[ ids[2] & mask[ numIds+0 ] ];
	verticesOut[0] = (const hkcdVertex&)a;
	verticesOut[1] = (const hkcdVertex&)b;
	verticesOut[2] = (const hkcdVertex&)c;
	HK_ASSERT( 0xA17A384B, numIds < 1 || verticesOut[0].getInt24W()==ids[0] );
	HK_ASSERT( 0xA17A384B, numIds < 2 || verticesOut[1].getInt24W()==ids[1] );
	HK_ASSERT( 0xA17A384B, numIds < 3 || verticesOut[2].getInt24W()==ids[2] );
#endif
}

HK_FORCE_INLINE void hknpConvexShape::getFaceInfo( const int faceId, hkVector4& planeOut, int& minAngleOut ) const
{
	hkVector4 v0 = m_vertices[(faceId >> 24) & 0xff];
	hkVector4 v1 = m_vertices[(faceId >> 16) & 0xff];
	hkVector4 v2 = m_vertices[(faceId >> 8) & 0xff];
	hkVector4 v3 = m_vertices[(faceId) & 0xff];

	// To allow quads to be non-flat, we report the area-weighted average normal.
	// This is computed by calculating the cross product of the diagonals. While mathematically identical
	// to adding the cross products of all four neighboring pairs of edges, this way is much faster.
	hkVector4 diag1; diag1.setSub(v2, v0);
	hkVector4 diag2; diag2.setSub(v3, v1);
	planeOut.setCross(diag1, diag2);
	planeOut.normalizeIfNotZero<3>();
	planeOut.setW(-planeOut.dot<3>(v0)); // Use v0 or avg vector instead?

	minAngleOut = 0; // TODO
}

HK_FORCE_INLINE int hknpConvexShape::getSupportingFace(
	hkVector4Parameter point, const hkcdGsk::Cache* gskCache, bool useB,
	hkVector4& planeOut, int& minAngleOut, hkUint32 &prevFaceId ) const
{
	const hkcdGsk::Cache::VertexId* vertexIds;
	int dim;
	if(!useB)
	{
		vertexIds = gskCache->getVertexIdsA();
		dim = gskCache->getDimA();
	}
	else
	{
		vertexIds = gskCache->getVertexIdsB();
		dim = gskCache->getDimB();
	}
	hkUint32 faceId = getSupportingFaceId(vertexIds, point, planeOut, getVertices(), getNumberOfVertices(), dim);
	getFaceInfo(faceId, planeOut, minAngleOut);
	faceId = revertToPrevFaceIdIfNeeded(this, faceId, planeOut, point, prevFaceId);
	return faceId;
}

#endif // HKNP_CONVEX_SHAPE_INL

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
