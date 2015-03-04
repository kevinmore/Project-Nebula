/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#ifndef HK_TEST_HULL_H
#define HK_TEST_HULL_H

#include <Common/Internal/ConvexHull/Deprecated/hkGeomHull.h>

#include <Common/Internal/ConvexHull/Deprecated/hkGeomConvexHullBuilder.h>


class hkGeomConvexHullTester
{
	public:
		static hkBool HK_CALL isValidHull( const hkGeomConvexHullTolerances&  tolerances, const hkVector4* initialVerts, int numInitialVertices, hkGeomHull& hull, hkArray<hkVector4>& usedVertices );
		static hkBool HK_CALL checkPlaneEquations( hkArray<hkVector4>& initialVerts, hkArray<hkVector4>& usedVertices, hkArray<hkVector4>& planeEquations, hkReal coplanar_tolerance );
		static hkBool HK_CALL isValidPlanarHull( hkArray<hkVector4>& initialVerts, hkGeomHull& hull, hkArray<hkVector4>& usedVertices, hkArray<hkVector4>& planeEquations, hkArray<hkGeomConvexHullBuilder::PlaneAndPoints>& tangentPlanes, hkReal coplanar_tolerance );

		static hkBool HK_CALL isValidNonPlanarHull( hkArray<hkVector4>& initialVerts, hkGeomHull& hull, hkArray<hkVector4>& usedVertices, hkArray<hkVector4>& planeEquations, hkArray<hkGeomConvexHullBuilder::PlaneAndPoints>& tangentPlanes, hkReal coplanar_tolerance );

		static hkBool HK_CALL findSameEdges( const hkGeomConvexHullBuilder::PlaneAndPoints& pp1, const hkGeomConvexHullBuilder::PlaneAndPoints& pp2, hkBool& found01, hkBool& found02, hkBool& found12 );

};

#endif //HK_TEST_HULL_H

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
