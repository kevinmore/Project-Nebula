/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_DEFAULT_EMS_MESH_H
#define HKNP_DEFAULT_EMS_MESH_H

#include <Physics/Physics/Collide/Shape/Composite/Mesh/Extern/hknpExternMeshShape.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>

/// Mesh interface for accessing external geometry stored as hkGeometry
class hknpDefaultExternMeshShapeMesh : public hknpExternMeshShape::Mesh
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS);

		hknpDefaultExternMeshShapeMesh(hkGeometry* geometry) : m_geometry(geometry) {}

		virtual int getNumTriangles() const HK_OVERRIDE
		{
			return m_geometry->m_triangles.getSize();
		}

		virtual void getTriangleVertices( int index, hkVector4* verticesOut ) const HK_OVERRIDE
		{
			m_geometry->getTriangle(index, verticesOut);
		}

		virtual hknpShapeTag getTriangleShapeTag( int index ) const HK_OVERRIDE
		{
			return (hknpShapeTag) m_geometry->m_triangles[index].m_material;
		}

	public:

		/// External geometry. Aligned for SPU access.
		HK_ALIGN16(hkRefPtr<hkGeometry> m_geometry);
};

#endif	// HKNP_DEFAULT_EMS_MESH_H

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
