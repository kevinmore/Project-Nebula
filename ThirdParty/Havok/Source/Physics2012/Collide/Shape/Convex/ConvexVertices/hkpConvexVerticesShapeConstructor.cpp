/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>

#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>

#include <Common/Base/Types/Geometry/hkStridedVertices.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesConnectivity.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesConnectivityUtil.h>
#include <Physics2012/Collide/Util/ShapeShrinker/hkpShapeShrinker.h>

#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>

hkpConvexVerticesShape::BuildConfig::BuildConfig()
{
	m_createConnectivity				=	false;
	m_shrinkByConvexRadius				=	true;
	m_useOptimizedShrinking				=	false;
	m_convexRadius						=	hkConvexShapeDefaultRadius;
	m_maxVertices						=	0;
	m_maxRelativeShrink					=	0.05f;
	m_maxShrinkingVerticesDisplacement	=	0.07f;
	m_maxCosAngleForBevelPlanes			=   -0.1f;
}

hkpConvexVerticesShape::hkpConvexVerticesShape(	const hkStridedVertices& vertices, const BuildConfig& config)
:	hkpConvexShape(HKCD_SHAPE_TYPE_FROM_CLASS(hkpConvexVerticesShape), config.m_convexRadius)
,	m_useSpuBuffer(false)
,	m_connectivity(HK_NULL)
{
	const hkgpConvexHull::Inputs	hullVertices	=	hkgpConvexHull::INTERNAL_VERTICES;
	hkgpConvexHull	hull;
	// Build the convex hull

	hkgpConvexHull::BuildConfig		b_config;	

	b_config.m_allowLowerDimensions					=	true;
	b_config.m_ensurePlaneEnclosing					=	true;
	b_config.m_checkForDegeneratedMassProperties	=	true;
	hull.build(vertices,b_config);
	
	if(hull.getDimensions()==-1)
	{
		HK_ERROR(0xC28C58E7,"Cannot create convex hull");
	}

	// Simplify the convex hull
	if(config.m_maxVertices>3 && hull.getDimensions()==3)
	{
		hull.decimateVertices(config.m_maxVertices,true);
	}

	// Build indices

	hull.buildIndices();

	// Simplify the convex hull by doing the following:
	// removing small triangles
	// removing vertices not referenced by connectivity (via hkgpConvexHull::generateIndexedFaces)

	/* EXP-1827 , disabling simplification as it used non-customizable absolute values
	hkgpConvexHull::SimplifyConfig	s_config;
	s_config.m_minArea						=	0.00001f;
	s_config.m_removeUnreferencedVertices	=	true;
	s_config.m_ensureContainment			=	true;
	s_config.m_maxVertices					=	config.m_maxVertices;
	s_config.m_forceRebuild					=	hull.getNumVertices()!=vertices.m_numVertices;

	hull.simplify(s_config);
	*/

	// Extract vertices and planes from the convex hull
	hkArray<hkVector4>		points;	

	hull.fetchPositions(hullVertices,points);
	hull.fetchPlanes(m_planeEquations);
	hull.fetchBevelPlanes( config.m_maxCosAngleForBevelPlanes, m_planeEquations);

	if(hull.getDimensions()==2)
	{
		hkVector4	frontPlane = hull.getProjectionPlane();
		hkVector4	backPlane; backPlane.setNeg<4>(frontPlane);
		m_planeEquations.pushBack(frontPlane);
		m_planeEquations.pushBack(backPlane);
	}

	// Make planes comply with unit tests.

	for(int i=0;i<m_planeEquations.getSize();++i)
	{
		m_planeEquations[i](3)+=HK_REAL_EPSILON;
	}
	
	// Build the shape
	hkStridedVertices		stridedPoints=points;
	copyVertexData(stridedPoints.m_vertices,stridedPoints.m_striding,stridedPoints.m_numVertices);

	// Build connectivity
	const bool	doShrink= config.m_shrinkByConvexRadius && config.m_convexRadius>0;
	if(config.m_createConnectivity || doShrink)
	{
		hkArray<hkUint8>				verticesPerFace;
		hkArray<hkUint16>				vertexIndices;
		hkpConvexVerticesConnectivity*	connectivity = new hkpConvexVerticesConnectivity();
		hull.generateIndexedFaces(hullVertices, verticesPerFace, vertexIndices, true);
		connectivity->m_numVerticesPerFace.swap(verticesPerFace);
		connectivity->m_vertexIndices.swap(vertexIndices);
		setConnectivity(connectivity, false);
		connectivity->removeReference();
	}
	// Shrinking
	if(doShrink)
	{
		hkpConvexVerticesShape* newShape=hkpShapeShrinker::shrinkConvexVerticesShape(this,config.m_convexRadius,config.m_maxRelativeShrink,config.m_maxShrinkingVerticesDisplacement,HK_NULL,config.m_useOptimizedShrinking);
		if(newShape && newShape!=this)
		{
			/* Swap 'newShape' and 'this' content.	*/ 
			hkAlgorithm::swap(newShape->m_aabbCenter		,	m_aabbCenter);
			hkAlgorithm::swap(newShape->m_aabbHalfExtents	,	m_aabbHalfExtents);
			hkAlgorithm::swap(newShape->m_numVertices		,	m_numVertices);
			hkAlgorithm::swap(newShape->m_connectivity		,	m_connectivity);
			hkAlgorithm::swap(newShape->m_radius			,	m_radius);
			newShape->m_rotatedVertices.swap(m_rotatedVertices);
			newShape->m_planeEquations.swap(m_planeEquations);
		}	
		else
		{
			hull.buildMassProperties();
			HK_WARN(0x638A1543,"Failed to shrink the shape (Volume: "<<hull.getVolume().getReal()<<")");
		}
		if(newShape) newShape->removeReference();
		//HK_REPORT("Requested convex radius: "<<config.m_convexRadius<<" , actual: "<<m_radius);
	}
	// Remove connectivity if not required
	if ( !config.m_createConnectivity && m_connectivity )
	{
		delete m_connectivity;
		m_connectivity = 0;
	}
}

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
