/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

// PCH
#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Common/Visualize/Shape/hkDisplayGeometry.h>

#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpShapeDisplayViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/hkpShapeDisplayBuilder.h>

// this
#include <Physics2012/Utilities/Collide/ShapeUtils/ShapeConverter/hkpShapeConverter.h>

hkGeometry* hkpShapeConverter::toSingleGeometry(const hkpShape* shape)
{
	
	hkInplaceArray<hkDisplayGeometry*, 8> displayGeometries;
	
	hkpShapeDisplayBuilder::hkpShapeDisplayBuilderEnvironment env;
	hkpShapeDisplayBuilder shapeBuilder(env);

	shapeBuilder.buildDisplayGeometries( shape, displayGeometries );

	// Lets concat it all together and put it in the same space
	hkGeometry* geomOut = new hkGeometry;

	for(int i = (displayGeometries.getSize() - 1); i >= 0; i--)
	{
		hkDisplayGeometry* disp = displayGeometries[i];
		disp->buildGeometry();

		if( disp->getGeometry() == HK_NULL )
		{
			HK_WARN_ALWAYS(0x6f7a9e3a, "Unable to build geometry from hkShape display geometry data.");
		}
		else
		{
			if( !disp->getGeometry()->isValid() )
			{
				HK_WARN_ALWAYS(0x6f7a9e3b, "Invalid geometry from hkShape display geometry data. Skipping it.");
			}
			else
			{
				// Append it 
				geomOut->appendGeometry( *disp->getGeometry(), disp->getTransform() );
			}
		}
		disp->removeReference();
	}

	// We only appended valid geometry, so the output had better be valid.
	const bool geomOutIsValid = geomOut->isValid();

	if (!geomOutIsValid || geomOut->m_vertices.getSize() <= 0)
	{
		HK_WARN_ALWAYS(0x6f7a9e3, "Appending geometries failed." );
		delete geomOut;
		return HK_NULL;
	}

	// Return final geom
	return geomOut;
}

hkGeometry* hkpShapeConverter::clone(const hkGeometry& geom)
{
    hkGeometry* out = new hkGeometry;
    out->m_vertices = geom.m_vertices;
    out->m_triangles = geom.m_triangles;
    return out;
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
