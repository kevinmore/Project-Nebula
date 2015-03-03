/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

#include <Geometry/Internal/Algorithms/RayCast/hkcdRayCastConvex.h>
#include <Geometry/Internal/Algorithms/SupportingVertex/hkcdSupportingVertex.h>
#include <Geometry/Internal/Algorithms/Gsk/hkcdGsk.h>

#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>

#include <Common/Base/Types/Geometry/hkStridedVertices.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesConnectivity.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesConnectivityUtil.h>
#include <Physics2012/Internal/Collide/Gjk/hkpGsk.h>

#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Common/Internal/GeometryProcessing/hkGeometryProcessing.h>
#include <Common/Base/Math/Vector/hkIntVector.h>

#if defined(HK_PLATFORM_SPU)
#include <Physics2012/Dynamics/World/Simulation/Multithreaded/Spu/hkpSpuConfig.h>
#include <Common/Base/Memory/PlatformUtils/Spu/SpuDmaCache/hkSpu4WayCache.h>
#endif

#if defined HK_COMPILER_MSVC
	// C4701: local variable 'lastNormal' and 'hitNormal' may be used without having been initialized
#	pragma warning(disable: 4701)
#endif

#if defined(HK_PLATFORM_SPU)

//#   define FETCH_VERTICES(fvEa,size) static_cast<const hkFourTransposedPoints*>( g_SpuCollideUntypedCache->getFromMainMemory( fvEa, size ))
	static inline const hkFourTransposedPoints* FETCH_VERTICES(const hkpConvexVerticesShape* thisObj, const void* fvEa, unsigned int size)
	{
		if ( thisObj->m_useSpuBuffer )
		{
			return static_cast<const hkFourTransposedPoints*>( fvEa );
		}
		return static_cast<const hkFourTransposedPoints*>( g_SpuCollideUntypedCache->getFromMainMemory( fvEa, size ));
	}

	static inline const hkFourTransposedPoints& FETCH_FOUR_VERTICES(const hkpConvexVerticesShape* thisObj, 
		const hkArray<hkFourTransposedPoints>& fvarray, unsigned int fvIndex)
	{
		if ( thisObj->m_useSpuBuffer )
		{
			return fvarray[fvIndex];
		}
		const unsigned foursPerLine = 256 / sizeof(hkFourTransposedPoints);
		const int cacheLine = fvIndex / foursPerLine;
		const int indexInCacheLine = fvIndex - cacheLine * foursPerLine;
		const int endClip = hkMath::max2( 0, int((cacheLine+1)*foursPerLine - fvarray.getSize()));
		int numFoursToFetch = foursPerLine - endClip;
		const hkFourTransposedPoints* fv = FETCH_VERTICES( thisObj, fvarray.begin() + cacheLine*foursPerLine, sizeof(hkFourTransposedPoints)*numFoursToFetch );
		return fv[indexInCacheLine];
	}

#else
#   define FETCH_VERTICES(thisObj, fvEa, size) (fvEa)
#   define FETCH_FOUR_VERTICES(thisObj, fvArray, fvIndex) fvArray[fvIndex]
#endif

#if !defined(HK_PLATFORM_SPU)

hkpConvexVerticesShape::hkpConvexVerticesShape(	hkReal radius )
:	hkpConvexShape( HKCD_SHAPE_TYPE_FROM_CLASS(hkpConvexVerticesShape), radius )
,	m_useSpuBuffer(false)
,	m_connectivity(HK_NULL)
{
}

hkpConvexVerticesShape::hkpConvexVerticesShape( const hkStridedVertices& vertsIn, const hkArray<hkVector4>& planeEquations, hkReal radius)
:	hkpConvexShape( HKCD_SHAPE_TYPE_FROM_CLASS(hkpConvexVerticesShape), radius )
,	m_useSpuBuffer(false)
,	m_connectivity(HK_NULL)
{
	HK_ASSERT2( 0x393378da, vertsIn.m_striding % hkSizeOf(float) == 0, "vertsIn.m_striding is not a multiple of hkSizeOf(float)." );
	HK_ASSERT2( 0x393378db, planeEquations.getSize() == 0 || planeEquations.getSize() >= 4,
		"Where plane equations are provided, at least 4 are required." );

	m_planeEquations = planeEquations;

	// copy all the data
	copyVertexData(vertsIn.m_vertices,vertsIn.m_striding,vertsIn.m_numVertices);
}

hkpConvexVerticesShape::hkpConvexVerticesShape(
		hkFourTransposedPoints* rotatedVertices, int numVertices,
		hkVector4* planes, int numPlanes,
		const hkAabb& aabb, hkReal radius )
	:	hkpConvexShape( HKCD_SHAPE_TYPE_FROM_CLASS(hkpConvexVerticesShape), radius ),
		m_rotatedVertices( rotatedVertices, HK_NEXT_MULTIPLE_OF(4, numVertices)/4, HK_NEXT_MULTIPLE_OF(4, numVertices)/4 ),
		m_numVertices(numVertices),
		m_useSpuBuffer(false),
		m_planeEquations(planes, numPlanes, numPlanes),
		m_connectivity(HK_NULL)
{
#ifdef HK_DEBUG
	{
		// Check validity of planes and vertices
		hkBool shapeOk = true;
		int checkedVertices = 0;
		for (int v = 0; v < m_rotatedVertices.getSize(); v++)
		{
			for (int rv = 0; rv < 4 && checkedVertices < m_numVertices; rv++, checkedVertices++)
			{
				hkReal* realPtr = reinterpret_cast<hkReal*>(&m_rotatedVertices[v].m_vertices[0]) + rv;
				hkVector4 originalVertex;
				originalVertex.set( realPtr[0], realPtr[4], realPtr[8] );

				for (int p = 0; p < m_planeEquations.getSize(); p++)
				{
					hkReal distFromPlane = ( originalVertex.dot<3>(m_planeEquations[p]) + m_planeEquations[p].getW() ).getReal();
					if (distFromPlane > 0.1f)
					{
						shapeOk = false;
						break;
					}
				}
			}
		}
		if (!shapeOk)
		{
			HK_WARN(0xad876bb4, "Vertices or planes are invalid");
		}
		HK_ASSERT2( 0x393378db, m_planeEquations.getSize() == 0 || m_planeEquations.getSize() >= 4,
			"Where plane equations are provided, at least 4 are required." );
	}
#endif

	m_aabbHalfExtents.setSub( aabb.m_max, aabb.m_min );
	m_aabbHalfExtents.mul( hkSimdReal::getConstant(HK_QUADREAL_INV_2) );
	m_aabbCenter.setAdd( aabb.m_min, aabb.m_max );
	m_aabbCenter.mul( hkSimdReal::getConstant(HK_QUADREAL_INV_2) );
}

hkpConvexVerticesShape::hkpConvexVerticesShape( hkFinishLoadedObjectFlag flag ) 
	: hkpConvexShape(flag)
	, m_rotatedVertices(flag)
	, m_planeEquations(flag) 
{ 
	if (flag.m_finishing)
	{
		setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpConvexVerticesShape));
	}
}

hkpConvexVerticesShape::~hkpConvexVerticesShape()
{
	if (m_connectivity)
	{
		m_connectivity->removeReference();
	}
}


void hkpConvexVerticesShape::setConnectivity(const hkpConvexVerticesConnectivity* connect, bool sort) 
{
	if ( connect ) 
	{ 
		connect->addReference(); 
	}

	if ( m_connectivity ) 
	{ 
		m_connectivity->removeReference(); 
	}

	m_connectivity = connect;

	if ( sort&& m_connectivity )
	{
		sortPlanes();
	}
}


void hkpConvexVerticesShape::getOriginalVertices( hkArray<hkVector4>& vertices ) const
{
	// Reserve space with some padding, resize correctly later
	vertices.setSize( m_rotatedVertices.getSize()*4 );
	hkFourTransposedPoints::getOriginalVertices( m_rotatedVertices.begin(), m_numVertices, static_cast<hkcdVertex*>(vertices.begin()) );
	vertices.setSize( m_numVertices );
}

void hkpConvexVerticesShape::copyVertexData(const hkReal* vertexIn, int byteStriding, int numVertices)
{
	
	HK_ASSERT2(0x601f8f0d, numVertices > 0, "numVertices <=0! must be > 0");

	m_numVertices = numVertices;

	int paddedSize = HK_NEXT_MULTIPLE_OF(4, numVertices);

	m_rotatedVertices.setSize( paddedSize / 4 );

	const hkReal* v = vertexIn;	
	const int numBatches = numVertices >> 2;
	for (int bi = 0; bi < numBatches; bi++)
	{
		hkVector4 vA, vB, vC, vD;
		
		vA.load<3,HK_IO_NATIVE_ALIGNED>(v);
		v = hkAddByteOffset( const_cast<hkReal*>(v), byteStriding );

		vB.load<3,HK_IO_NATIVE_ALIGNED>(v);
		v = hkAddByteOffset( const_cast<hkReal*>(v), byteStriding );

		vC.load<3,HK_IO_NATIVE_ALIGNED>(v);
		v = hkAddByteOffset( const_cast<hkReal*>(v), byteStriding );

		vD.load<3,HK_IO_NATIVE_ALIGNED>(v);
		v = hkAddByteOffset( const_cast<hkReal*>(v), byteStriding );

		m_rotatedVertices[bi].set(vA, vB, vC, vD);
	}

	// Last batch
	const int numRemaining = numVertices - (numBatches << 2);
	if ( numRemaining )
	{
		// Get remaining vertices
		hkVector4 verts[4];
		for (int i = 0; i < numRemaining; i++)
		{
			verts[i].load<3,HK_IO_NATIVE_ALIGNED>(v);
			v = hkAddByteOffset(const_cast<hkReal*>(v), byteStriding);
		}

		// Fill rest with the last vertex
		v = hkAddByteOffset( const_cast<hkReal*>(v), -byteStriding );
		for (int i = numRemaining; i < 4; i++)
		{
			verts[i].load<3,HK_IO_NATIVE_ALIGNED>(v);
		}

		m_rotatedVertices[numBatches].set(verts[0], verts[1], verts[2], verts[3]);
	}

	hkAabb aabb;
	hkAabbUtil::calcAabb( vertexIn, numVertices, byteStriding, aabb );

	aabb.getHalfExtents( m_aabbHalfExtents );
	aabb.getCenter( m_aabbCenter );
}

void hkpConvexVerticesShape::getFirstVertex(hkVector4& v) const
{
	m_rotatedVertices[0].extract(0, v);
}

#endif //HK_PLATFORM_SPU

#if !defined(HK_PLATFORM_SPU)
const hkSphere* hkpConvexVerticesShape::getCollisionSpheres(hkSphere* sphereBuffer) const
{
	hkSphere* s = sphereBuffer;
	const hkFourTransposedPoints* fv = m_rotatedVertices.begin();
	int i = m_numVertices - 1;

	hkVector4 thisRadius; thisRadius.setAll( m_radius );
	for ( ; i >=3 ; i-=4)
	{
		hkVector4 v0, v1, v2, v3;
		fv->extractWithW(thisRadius, v0, v1, v2, v3);

		s[0].setPositionAndRadius(v0);
		s[1].setPositionAndRadius(v1);
		s[2].setPositionAndRadius(v2);
		s[3].setPositionAndRadius(v3);

		s += 4;
		fv++;
	}

	if( i >= 0 )
	{
		hkVector4 v0, v1, v2, v3;
		fv->extractWithW(thisRadius, v0, v1, v2, v3);

		switch( i )
		{
			case 2: s[2].setPositionAndRadius(v2); // fallthrough
			case 1: s[1].setPositionAndRadius(v1); // fallthrough
			case 0: s[0].setPositionAndRadius(v0);
		}
	}

	return sphereBuffer;
}
#else
const hkSphere* hkpConvexVerticesShape::getCollisionSpheres(hkSphere* sphereBuffer) const
{
	hkSphere* s = sphereBuffer;

	unsigned int totalVertices = m_numVertices;
	unsigned int maxVertices = (256/sizeof(hkFourTransposedPoints))*4;

	unsigned char *currentEA = (unsigned char *) &(m_rotatedVertices[0]);

	int i;
	const hkFourTransposedPoints* fv;
	hkVector4 thisRadius; thisRadius.setAll(m_radius);
	do
	{
		unsigned int currentVertices = hkMath::min2(totalVertices, maxVertices);
		unsigned int dmaSize = HK_NEXT_MULTIPLE_OF(4, currentVertices) * (sizeof(hkFourTransposedPoints)/4);
		fv = FETCH_VERTICES(this, currentEA,dmaSize);
		totalVertices-=currentVertices;
		currentEA+=dmaSize;

		i = currentVertices - 1;
		for ( ; i >=3 ; i-=4)
		{
		
			hkVector4 v0, v1, v2, v3;
			fv->extractWithW(thisRadius, v0, v1, v2, v3);

			s[0].setPositionAndRadius(v0);
			s[1].setPositionAndRadius(v1);
			s[2].setPositionAndRadius(v2);
			s[3].setPositionAndRadius(v3);

			s += 4;
			fv++;
		}
	} while(totalVertices);

	{
		hkVector4 v0, v1, v2, v3;
		fv->extractWithW(thisRadius, v0, v1, v2, v3);

		switch( i )
		{
			case 2: s[2].setPositionAndRadius(v2); // fallthrough
			case 1: s[1].setPositionAndRadius(v1); // fallthrough
			case 0: s[0].setPositionAndRadius(v0);
		}
	}

	return sphereBuffer;
}
#endif

void hkpConvexVerticesShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	hkAabbUtil::calcAabb( localToWorld, m_aabbHalfExtents, m_aabbCenter, hkSimdReal::fromFloat(tolerance + m_radius),  out );
}

void hkpConvexVerticesShape::getSupportingVertex(hkVector4Parameter direction, hkcdVertex& supportingVertexOut) const
#if defined(HK_PLATFORM_SPU)
{
	//HK_INTERNAL_TIMER_BEGIN( "support", HK_NULL );
	hkVector4 bestDot = hkVector4::getConstant<HK_QUADREAL_MINUS_MAX>();
	
	static const hkQuadReal _curIndices = HK_QUADREAL_CONSTANT(0.5f,1+0.5f,2+0.5f,3+0.5f);
	hkVector4 curIndices; curIndices.m_quad = _curIndices;

	hkVector4 stepIndices = hkVector4::getConstant<HK_QUADREAL_4>();
	hkVector4 bestIndices;

	int totalNum = m_rotatedVertices.getSize() << 2;
	const int maxVerticesPerFetch = (256/sizeof(hkFourTransposedPoints))*4;
	unsigned char  *currentEA = (unsigned char *) &(m_rotatedVertices[0]);

	hkVector4 d0; d0.setBroadcast<0>(direction);
	hkVector4 d1; d1.setBroadcast<1>(direction);
	hkVector4 d2; d2.setBroadcast<2>(direction);

	do
	{
		int numThisFetch = hkMath::min2(totalNum, maxVerticesPerFetch);
		unsigned int dmaSize = HK_NEXT_MULTIPLE_OF(4,numThisFetch) * (sizeof(hkFourTransposedPoints)/4);

		const hkFourTransposedPoints* fv = FETCH_VERTICES(this, currentEA,dmaSize);
		
		totalNum -= numThisFetch;
		currentEA += dmaSize;

		HK_ASSERT( 0x4c5c7d57, numThisFetch > 0 ); // must have some elements or bestIndices is uninitialized

		// get max dots four at a time
		for ( int i = 0; i < numThisFetch; fv++, i+=4 )
		{
			hkVector4 curDot;

			// calculate the dot product for four vertices
			{
				hkVector4 x; x.setMul( d0, fv->m_vertices[0] );
				hkVector4 y; y.setMul( d1, fv->m_vertices[1] );
				curDot.setAdd( x,y );
				hkVector4 z; z.setMul( d2, fv->m_vertices[2] );
				curDot.add( z );
			}

			hkVector4Comparison comp = bestDot.less( curDot );
			bestDot.setSelect(comp, curDot, bestDot);
			bestIndices.setSelect(comp, curIndices, bestIndices);
			curIndices.setAdd( curIndices, stepIndices );
		}
	} while( totalNum );

	// find the best of the 4 we have
	int bestIndex4 = bestDot.getIndexOfMaxComponent<4>();

	// extract vertex
	int vertexId = (int)bestIndices( bestIndex4 );
	HK_ASSERT(0x778ba506, vertexId >= 0 && vertexId < 4*m_rotatedVertices.getSize() );
	{
		const hkFourTransposedPoints& f = FETCH_FOUR_VERTICES(this, m_rotatedVertices, unsigned(vertexId)>>2);
		int a = vertexId & 3;
#if !defined(HK_PLATFORM_SPU)
		supportingVertexOut(0) = f.m_x(a);
		supportingVertexOut(1) = f.m_y(a);
		supportingVertexOut(2) = f.m_z(a);
		supportingVertexOut.setInt24W( vertexId );
#else
		hkVector4 v;
		v(0) = f.m_vertices[0](a);
		v(1) = f.m_vertices[1](a);
		v(2) = f.m_vertices[2](a);
		v.setInt24W( vertexId );
		(hkVector4&)supportingVertexOut = v;
#endif


	}
}
#elif (HK_CONFIG_SIMD==HK_CONFIG_SIMD_ENABLED) && (!defined HK_PLATFORM_LINUX) // This code does not compile on Linux
{
	const hkArray<hkFourTransposedPoints>& transposedVerts = m_rotatedVertices;
	hkcdSupportingVertexPoints(transposedVerts.begin(), transposedVerts.getSize(), direction, &supportingVertexOut);
}
#else
{
	//HK_INTERNAL_TIMER_BEGIN( "support", HK_NULL );
	const hkFourTransposedPoints* fv;

	hkVector4 bestDot;

	int vertexId = 0;

//		// initialize the best distance based on the old dist
//	if (1){
//		vertexId = vertexIdCache;
//		HK_ASSERT2(0x516a1efb,  vertexId>=0 && vertexId < m_rotatedVertices.getSize() * 4, "vertexIdCache invalid (try 0)");
//		fv = &m_rotatedVertices[vertexId>>2];
//		int a = vertexId & 3;
//
//		hkReal dot = fv->m_x(a) * direction(0) + fv->m_y(a) * direction(1) + fv->m_z(a) * direction(2);
//		bestDot.setAll(dot);
//	}
//	else
	{
		bestDot = hkVector4::getConstant<HK_QUADREAL_MINUS_MAX>();
	}


	fv = &(m_rotatedVertices[0]);
	int maxNum = m_rotatedVertices.getSize() << 2;
	for ( int i = 0; i < maxNum; fv++, i+=4 )
	{
		hkVector4 dot;

		// calculate the dot product for four vertices
		{
			hkVector4 x; x.setMul( direction.getComponent<0>(), fv->m_vertices[0] );
			hkVector4 y; y.setMul( direction.getComponent<1>(), fv->m_vertices[1] );
			dot.setAdd( x,y );
			hkVector4 z; z.setMul( direction.getComponent<2>(), fv->m_vertices[2] );
			dot.setAdd( dot,z );
		}

		hkVector4ComparisonMask::Mask flags = bestDot.less( dot ).getMask();
		//all values are less then current best 'signed distance'
		if ( flags == hkVector4ComparisonMask::MASK_NONE )
		{
			continue;
		}
		hkSimdReal d;
		switch (flags)
		{
#define TWO_CASE(a,b) if ( dot.getComponent<(a)>() > dot.getComponent<(b)>() ) { d = dot.getComponent<(a)>(); vertexId = i + a; } else { d = dot.getComponent<(b)>(); vertexId = i + b; } break;
#define THREE_CASE(a,b,c) if ( dot.getComponent<(a)>() > dot.getComponent<(b)>() ) { TWO_CASE(a, c); }else{ TWO_CASE( b, c ) };

			case hkVector4ComparisonMask::MASK_NONE:
			case hkVector4ComparisonMask::MASK_X:
				d = dot.getComponent<0>();
				vertexId = i;
				break;
			case hkVector4ComparisonMask::MASK_Y:
				d = dot.getComponent<1>();
				vertexId = i + 1;
				break;
			case hkVector4ComparisonMask::MASK_Z:
				d = dot.getComponent<2>();
				vertexId = i + 2;
				break;
			case hkVector4ComparisonMask::MASK_W:
				d = dot.getComponent<3>();
				vertexId = i + 3;
				break;
/*
			// save code space by handling these cases in default:

			case hkVector4ComparisonMask::MASK_XY:
				TWO_CASE(0,1);
			case hkVector4ComparisonMask::MASK_XZ:
				TWO_CASE(0,2);
			case hkVector4ComparisonMask::MASK_XW:
				TWO_CASE(0,3);
			case hkVector4ComparisonMask::MASK_YZ:
				TWO_CASE(1,2);
			case hkVector4ComparisonMask::MASK_YW:
				TWO_CASE(1,3);
			case hkVector4ComparisonMask::MASK_ZW:
				TWO_CASE(2,3);
			case hkVector4ComparisonMask::MASK_XYZ:
				THREE_CASE(0,1,2);
			case hkVector4ComparisonMask::MASK_XYW:
				THREE_CASE(0,1,3);
*/
			case hkVector4ComparisonMask::MASK_XZW:
threeCase023:
				THREE_CASE(0,2,3);
			case hkVector4ComparisonMask::MASK_YZW:
threeCase123:
				THREE_CASE(1,2,3);

			default:
			case hkVector4ComparisonMask::MASK_XYZW:
				if ( dot.getComponent<0>() > dot.getComponent<1>() )
				{
					goto threeCase023;
				}
				else
				{
					goto threeCase123;
				}
		}
		bestDot.setAll(d);
	}
	{
		fv = &(m_rotatedVertices[unsigned(vertexId)>>2]);
		fv->extract(vertexId & 3, supportingVertexOut);
		supportingVertexOut.setInt24W( vertexId );
	}
	//HK_INTERNAL_TIMER_END();
}
#endif

void hkpConvexVerticesShape::convertVertexIdsToVertices(const hkpVertexId* ids, int numIds, hkcdVertex* verticesOut) const
{
	for (int i = numIds-1; i>=0; i--)
	{
		int vertexId = ids[0];
		const hkFourTransposedPoints& f = FETCH_FOUR_VERTICES(this, m_rotatedVertices, vertexId>>2);
		int a = vertexId & 3;
#if !defined(HK_PLATFORM_SPU)
		hkVector4& v = verticesOut[0];
		f.extract(a, v);
		v.setInt24W( vertexId );
#else
		hkVector4 v;
		f.extract(a, v);
		v.setInt24W( vertexId );
		(hkVector4&)verticesOut[0] = v;
#endif
		verticesOut++;
		ids++;
	}
}

void hkpConvexVerticesShape::getCentre(hkVector4& centreOut) const
{
	centreOut = m_aabbCenter;
}

static inline const hkVector4& ConvexVerticesShape_FetchVectorFromPpuArray( const hkVector4* base, int index )
{
#if defined(HK_PLATFORM_SPU)
	// This works because hkVector4s are not split across 256 byte boundaries in main memory
	HK_ASSERT2( 0x45654345, ( (hkUlong)base & 0xf ) == 0, "Array must be aligned");
	hkUlong vecAddr = (hkUlong)(&base[ index ]);
	const hkUlong mask = static_cast<hkUlong>(HK_SPU_AGENT_SECTOR_JOB_MAX_UNTYPED_CACHE_LINE_SIZE - 1);
	const char * alignedBase = reinterpret_cast<const char*> ( g_SpuCollideUntypedCache->getFromMainMemory( (void*)(vecAddr & ~mask), HK_SPU_AGENT_SECTOR_JOB_MAX_UNTYPED_CACHE_LINE_SIZE ) );
	return *reinterpret_cast<const hkVector4*>( hkAddByteOffsetConst( alignedBase, vecAddr & mask ) );
#else
	return base[index];
#endif
}

// Boundary coordinate sign bit meanings for the "AND" of the 'outcodes'
// sign (b)	sign (t)
// 0		0		whole segment inside box
// 1		0		b is outside box, t is in
// 0		1		b is inside box, t is out
// 1		1		whole segment outside box

// ray-convex (and ray-box) intersection with ideas from 'A trip down the graphics pipeline', Jim Blinn
// if ray starts within the box, no intersection is returned
// return 1 for success, 0 for no intersection. when success then hitpoint and hitNormal is filled

hkBool hkpConvexVerticesShape::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results) const
{
	// compute outcodes for begin and endpoint of the ray against every plane
	//	logical AND of the outcodes must be 0, if not, then there exists a face for which
	//	both the begin and endpoint of the ray is outside

	if( m_planeEquations.getSize() == 0 )
	{
		HK_TIMER_BEGIN("convexVertCastRayGSK", HK_NULL);

		const hkFourTransposedPoints* rotatedVerticesPtr = m_rotatedVertices.begin();

		// On SPU we must DMA the vertices first
#ifdef HK_PLATFORM_SPU
		const int rotatedVerticesSize = m_rotatedVertices.getSize();
		hkLocalArray<hkFourTransposedPoints> spuRotatedVertices(rotatedVerticesSize);
		{
			const int rotatedVerticesMemorySize = rotatedVerticesSize * sizeof(hkFourTransposedPoints);
			hkSpuDmaManager::getFromMainMemory(spuRotatedVertices.begin(), m_rotatedVertices.begin(), rotatedVerticesMemorySize, hkSpuDmaManager::READ_COPY);
			hkSpuDmaManager::waitForAllDmaCompletion();
			HK_SPU_DMA_PERFORM_FINAL_CHECKS(m_rotatedVertices.begin(), spuRotatedVertices.begin(), rotatedVerticesMemorySize);
		}

		rotatedVerticesPtr = spuRotatedVertices.begin();
#endif

		// Initialize ray-cast input
		hkcdGsk::RayCastInput gskInput(m_radius);
		gskInput.m_from	= input.m_from;
		gskInput.m_direction.setSub(input.m_to, input.m_from);

		hkcdGsk::RayCastOutput gskOutput;
		gskOutput.m_fractionInOut.setFromFloat(results.m_hitFraction);
		gskOutput.m_normalOut.setZero();

		// Run the ray-cast
		hkBool hit = hkcdGsk::rayCast(rotatedVerticesPtr, m_numVertices, gskInput, gskOutput);
		if ( hit )
		{
			results.m_normal = gskOutput.m_normalOut;
			results.m_hitFraction = gskOutput.m_fractionInOut.getReal();
			results.setKey( HK_INVALID_SHAPE_KEY );
		}

		HK_TIMER_END();
		return hit;
	}

	HK_TIMER_BEGIN("convexVertCastRayPlaneEq", HK_NULL);

	
	{
		// Make sure the number of planes fits the SPU maximum buffer size
		const int numEquations = m_planeEquations.getSize();
		hkVector4* planeEquationsPtr = m_planeEquations.begin();

		// On SPU we must DMA the plane equations first
#ifdef HK_PLATFORM_SPU
		hkLocalArray<hkVector4> spuPlaneEqns(numEquations);
		HK_SPU_STACK_POINTER_CHECK();
		{
			// Transfer equations from PPU
			const int dataSize = numEquations * sizeof(hkVector4);
			const void* ppuAddress = m_planeEquations.begin();
			void* spuAddress = spuPlaneEqns.begin();
			hkSpuDmaManager::getFromMainMemory(spuAddress, ppuAddress, dataSize, hkSpuDmaManager::READ_COPY);
			hkSpuDmaManager::waitForAllDmaCompletion();
			HK_SPU_DMA_PERFORM_FINAL_CHECKS(ppuAddress, spuAddress, dataSize);

			planeEquationsPtr = (hkVector4*)spuAddress;
		}
#endif
		
		// Cast ray
		hkcdRay ray;
		ray.setEndPoints( input.m_from, input.m_to );

		hkSimdReal fractionInOut = hkSimdReal::fromFloat(results.m_hitFraction);
		hkVector4 normalOut;

		hkBool32 ret = hkcdRayCastConvex(ray, planeEquationsPtr, numEquations, &fractionInOut, &normalOut, hkcdRayQueryFlags::NO_FLAGS);

		if ( ret )
		{
			results.m_normal = normalOut;
			results.m_hitFraction = fractionInOut.getReal();
			results.setKey( HK_INVALID_SHAPE_KEY );
		}

		HK_TIMER_END();
		return ret ? true : false;
	}
}

const hkArray<hkVector4>& hkpConvexVerticesShape::getPlaneEquations() const
{
	return m_planeEquations;
}

void hkpConvexVerticesShape::setPlaneEquations( const hkArray<hkVector4>& planes )
{
	HK_ASSERT2( 0x393378db, planes.getSize() == 0 || planes.getSize() >= 4,
		"Where plane equations are provided, at least 4 are required." );
	m_planeEquations = planes;

#if 0 && defined(HK_DEBUG)
	for(int i=0; i<planes.getSize(); ++i)
	{
		hkcdVertex	sv; getSupportingVertex(planes[i],sv);
		hkReal		d2p = planes[i].dot4xyz1(sv).getReal();
		if(d2p>0.001f) HK_WARN(0x2442DEAD, "Invalid plane("<<i<<")");
	}
#endif

#ifndef HK_PLATFORM_SPU
	if ( m_connectivity )	
	{
		HK_ASSERT( 0xddfe4534, planes.getSize() >= m_connectivity->getNumFaces());
		sortPlanes();
	}
#endif
}

void hkpConvexVerticesShape::transformVerticesAndPlaneEquations( const hkTransform& t )
{
#if !defined(HK_PLATFORM_SPU)

	hkLocalArray<hkVector4> vertices( getNumCollisionSpheres() );
	getOriginalVertices( vertices );

	hkVector4Util::transformPoints( t, vertices.begin(), vertices.getSize(), vertices.begin() );
	copyVertexData( &vertices.begin()[0](0), sizeof(hkVector4), vertices.getSize() );

	hkVector4Util::transformPlaneEquations( t, m_planeEquations.begin(), m_planeEquations.getSize(), m_planeEquations.begin() );

#else

	HK_ASSERT2(0x5051fcd9,0,"Not implemented on SPU");

#endif
}

#ifndef HK_PLATFORM_SPU
void hkpConvexVerticesShape::sortPlanes()
{
	HK_ASSERT( 0xdd76ed11, m_connectivity != HK_NULL );

	hkArray<hkVector4> planes; planes = m_planeEquations;
	m_planeEquations.clear();

	hkLocalArray< hkVector4 > vertices( m_numVertices );
	getOriginalVertices( vertices );

	int faceStart = 0;
	for ( int i = 0; i < m_connectivity->getNumFaces(); ++i )
	{
		if ( planes.isEmpty() )
		{
			break;
		}
		int numFaceIndices = m_connectivity->m_numVerticesPerFace[ i ];
		if ( numFaceIndices < 3 )
		{
			continue;	// dd.todo.aaa this will skip one plane equation / not acceptable
		}

		int idx0 = m_connectivity->m_vertexIndices[ faceStart + 0 ];
		int idx1 = m_connectivity->m_vertexIndices[ faceStart + 1 ];
		int idx2 = m_connectivity->m_vertexIndices[ faceStart + 2 ];
		faceStart += numFaceIndices;

		hkVector4 v0 = vertices[ idx0 ];
		hkVector4 v1 = vertices[ idx1 ];
		hkVector4 v2 = vertices[ idx2 ];

		hkVector4 e1, e2;
		e1.setSub( v1, v0 );
		e2.setSub( v2, v0 );

		hkVector4 n;
		n.setCross( e1, e2 );
	
		if ( n.normalizeIfNotZero<3>() )
		{
			int bestPlane = 0;
			hkSimdReal bestDot = planes[ 0 ].dot<3>( n );

			for ( int k = 1; k < planes.getSize(); ++k )
			{
				hkSimdReal dot = planes[ k ].dot<3>( n );
				if ( dot > bestDot )
				{
					bestDot = dot;
					bestPlane = k;
				}
			}

			m_planeEquations.pushBackUnchecked( planes[ bestPlane ] );
			planes.removeAt( bestPlane );
		}
		else
		{
			m_planeEquations.pushBackUnchecked( planes[ 0 ] );
		}
		
	}

	// Push back remaining bevel planes
	if ( planes.getSize() > 0 )
	{
		m_planeEquations.insertAt( m_planeEquations.getSize(), &planes[ 0 ], planes.getSize() );
	}
	HK_ASSERT2( 0xdd34edde, m_planeEquations.getSize() >= m_connectivity->m_numVerticesPerFace.getSize(), "You do not have enough plane equations" );
}
#endif //!SPU

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
