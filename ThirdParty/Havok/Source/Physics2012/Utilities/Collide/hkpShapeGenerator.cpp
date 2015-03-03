/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Collide/hkpShapeGenerator.h>
#include <Common/Internal/ConvexHull/hkGeometryUtility.h>

#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesConnectivityUtil.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>

#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>


hkpConvexVerticesShape* HK_CALL hkpShapeGenerator::createRandomConvexVerticesShape(	const hkVector4& minbox,
																					const hkVector4& maxbox,
																					int numvert, 
																					hkPseudoRandomGenerator *generator, 
																					Flags flags )
{
	HK_ASSERT(0x4028019b,  minbox.allLess<3>(maxbox));
	hkArray<hkVector4> verts(numvert);
	for(int i = 0; i < numvert; ++i)
	{
		hkReal v0 = generator->getRandRange( minbox(0), maxbox(0) );
		hkReal v1 = generator->getRandRange( minbox(1), maxbox(1) );
		hkReal v2 = generator->getRandRange( minbox(2), maxbox(2) );
		verts[i].set(v0,v1,v2);
	}

	hkStridedVertices stridedVerts;
	{
		stridedVerts.m_numVertices = numvert;
		stridedVerts.m_striding = sizeof(hkVector4);
		stridedVerts.m_vertices = &(verts[0](0));
	}


	if ( flags == NO_PLANE_EQUATIONS )
	{
		hkArray<hkVector4> dummyPlaneEquations;

		return new hkpConvexVerticesShape( stridedVerts, dummyPlaneEquations);
	}
	return new hkpConvexVerticesShape(verts);	
}


hkpConvexVerticesShape* HK_CALL hkpShapeGenerator::createRandomConvexVerticesShapeWithThinTriangles(	const hkVector4& minbox,
																									const hkVector4& maxbox,
																									int numvert, 
																									float minEdgeLen, 
																									hkPseudoRandomGenerator *generator, 
																									Flags flags )
{
	HK_ASSERT(0x79af13e8,  minbox.allLess<3>(maxbox) );
	hkInplaceArrayAligned16<hkVector4,48> verts(numvert+1);
	for(int i = 0; i < numvert; ++i)
	{
		{
			for(int j = 0; j < 3; ++j)
			{
				verts[i](j) = generator->getRandRange( minbox(j), maxbox(j) );
			}
		}
		{
			for(int j = 0; j < 3; ++j)
			{
				verts[i+1](j) = verts[i](j) + generator->getRandRange( -minEdgeLen, minEdgeLen );
			}
		}
		i++;
	}

	hkStridedVertices stridedVerts;
	{
		stridedVerts.m_numVertices = numvert;
		stridedVerts.m_striding = sizeof(hkVector4);
		stridedVerts.m_vertices = &(verts[0](0));
	}


	if ( flags == NO_PLANE_EQUATIONS )
	{
		hkArray<hkVector4> dummyPlaneEquations;
		return new hkpConvexVerticesShape( stridedVerts, dummyPlaneEquations);
	}
	return new hkpConvexVerticesShape(verts);
}

static HK_ALIGN_REAL( const hkReal vertexSignArray[8][4] ) = {
	{ 1, 1, 1, 0 },	// zyx = 000
	{-1, 1, 1, 0 },	// zyx = 001  
	{ 1,-1, 1, 0 },	// zyx = 010
	{-1,-1, 1, 0 },	// zyx = 011
	{ 1, 1,-1, 0 },	// zyx = 100
	{-1, 1,-1, 0 }, // zyx = 101
	{ 1,-1,-1, 0 }, // zyx = 110
	{-1,-1,-1, 0 }	// zyx = 111
};

hkpConvexShape* HK_CALL hkpShapeGenerator::createConvexShape( const hkVector4& extents, ShapeType type, hkPseudoRandomGenerator *generator )
{
	if ( type == RANDOM )
	{
		type = ShapeType( int(generator->getRandRange( RANDOM+1, SHAPE_MAX ) ) );
	}

	switch(type)
	{
	case	BOX:
		{
			hkpConvexShape* box =  new hkpBoxShape( extents );
			box->setRadius(0.f);
			return box;
		}
	case	SPHERE:
		{
			const hkReal minLen = extents.horizontalMin<3>().getReal();
			return new hkpSphereShape( minLen );
		}

	case CAPSULE:
		{
			const hkSimdReal radius = extents.horizontalMin<3>() * hkSimdReal_Inv2;
			hkVector4 A; A.setSub( extents, radius );
			hkVector4 B; B.setNeg<4>(A);
			return new hkpCapsuleShape( A,B,radius.getReal() );
		}

	case	TRIANGLE:
		{
			int i;
			if ( extents.getComponent<0>() > extents.getComponent<1>() )
			{
				i = 0;
			}
			else
			{
				i = 1;
			}
			hkpTriangleShape* shape = new hkpTriangleShape();
			shape->setVertex<0>(extents);
			hkVector4 nextents; nextents.setNeg<4>(extents);
			shape->setVertex<1>( nextents );
			nextents = extents;
			nextents(i) *= -1;
			shape->setVertex<2>(nextents);
			return shape;
		}

		/*
	case THIN_TRIANGLE:
		{
			int i;
			if ( extents(0) > extents(1) )
			{
				i = 0;
			}
			else
			{
				i = 1;
			}
			hkpTriangleShape* shape = new hkpTriangleShape();
			shape->getVertex(0) = extents;
			shape->getVertex(1).setNeg4( extents );
			shape->getVertex(2) = extents;
			shape->getVertex(2)(i) *= 0.99f;
			return shape;
		}
		*/

	case	CONVEX_VERTICES:
		{
			hkVector4 negExtents; negExtents.setNeg<4>( extents );
			return createRandomConvexVerticesShape( negExtents, extents, 30, generator, NONE );
		}
	case	CONVEX_VERTICES_BOX:
		{
		    hkVector4 scale; scale.setMul( hkSimdReal_Inv2, extents );
    
		    hkVector4 vertices[ 8 ];
		    for ( int i = 0; i < 8; ++i )
		    {
			    vertices[ i ].setMul( scale, reinterpret_cast< const hkVector4& >( *vertexSignArray[ i ] ) );
		    }
			return new hkpConvexVerticesShape(hkStridedVertices(vertices,8));
		}

	default:
		HK_ASSERT2(0x76e86a15,  0, "unknown shape type" );
		return HK_NULL;
	}
}

hkpConvexVerticesShape* HK_CALL hkpShapeGenerator::createConvexVerticesBox( const hkVector4& halfExtents, hkReal convexRadius )
{
	hkVector4 vertices[ 8 ];
	for ( int i = 0; i < 8; ++i )
	{
		vertices[ i ].setMul( halfExtents, reinterpret_cast< const hkVector4& >( *vertexSignArray[ i ] ) );
	}
	hkpConvexVerticesShape::BuildConfig buildConfig;
	{
		buildConfig.m_convexRadius = convexRadius;
	}
	return new hkpConvexVerticesShape( hkStridedVertices( vertices, 8 ), buildConfig );
}

const char* HK_CALL hkpShapeGenerator::getShapeTypeName( ShapeType type )
{
	switch(type)
	{
	case	BOX: return "BOX";
	case SPHERE: return "SPHERE";
	case CAPSULE: return "CAPSULE";
	case TRIANGLE: return "TRIANGLE";
	//case THIN_TRIANGLE: return "THIN_TRIANGLE";
	case CONVEX_VERTICES: return "CONVEX_VERTICES";
	case CONVEX_VERTICES_BOX: return "CONVEX_VERTICES_BOX";
	default: return "Unknown";
	}
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
