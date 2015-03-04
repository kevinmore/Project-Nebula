/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

HK_FORCE_INLINE hkcdPlanarGeometry::PolygonId hkcdPlanarGeometry::addPolygon(PlaneId supportPlane, hkUint32 material, int numBounds)
{
	return m_polys->alloc(supportPlane, material, numBounds);
}

HK_FORCE_INLINE const hkcdPlanarGeometry::Polygon& hkcdPlanarGeometry::getPolygon(PolygonId polyId) const
{
	return m_polys->getPolygon(polyId);
}

HK_FORCE_INLINE hkcdPlanarGeometry::Polygon& hkcdPlanarGeometry::accessPolygon(PolygonId polyId)
{
	return m_polys->accessPolygon(polyId);
}

//
//	Computes the number of boundary planes

HK_FORCE_INLINE int hkcdPlanarGeometry::getNumBoundaryPlanes(PolygonId polyId) const
{
	return m_polys->getNumBoundaryPlanes(polyId);
}

//
//	Returns the collection of polygons

HK_FORCE_INLINE const hkcdPlanarGeometryPolygonCollection& hkcdPlanarGeometry::getPolygons() const	{	return *m_polys;	}
HK_FORCE_INLINE hkcdPlanarGeometryPolygonCollection& hkcdPlanarGeometry::accessPolygons()			{	return *m_polys;	}

//
//	Gets the plane collection associated with this geometry

HK_FORCE_INLINE const hkcdPlanarGeometryPlanesCollection* hkcdPlanarGeometry::getPlanesCollection() const	{	return m_planes;	}
HK_FORCE_INLINE hkcdPlanarGeometryPlanesCollection* hkcdPlanarGeometry::accessPlanesCollection()			{	return m_planes;	}

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
