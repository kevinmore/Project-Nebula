/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Geometry/hkpGeometryConverter.h>

#include <Common/Base/Types/Geometry/hkGeometry.h>

#include <Common/Visualize/Shape/hkDisplayGeometry.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>
#include <Common/Visualize/hkProcessFactory.h>

#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

#include <Common/Visualize/hkVisualDebugger.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/Shape/hkDisplayConvex.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/hkpShapeDisplayBuilder.h>


/// Creates a geometry from a Havok physics world
void hkpGeometryConverter::createSingleGeometryFromWorld(const hkpWorld& world, hkGeometry& geomOut, 
																	hkBool useFixedBodiesOnly, hkBool getMaterialFromUserData)
{
	world.markForRead();

	if (!useFixedBodiesOnly)
	{
		// get all the active entities from the active simulation islands
		{
			const hkArray<hkpSimulationIsland*>& activeIslands = world.getActiveSimulationIslands();

			for(int i = 0; i < activeIslands.getSize(); i++)
			{
				const hkArray<hkpEntity*>& activeEntities = activeIslands[i]->getEntities();
				for(int j = 0; j < activeEntities.getSize(); j++)
				{
					appendGeometryFromRigidBody(*static_cast<hkpRigidBody*>( activeEntities[j] ), geomOut, getMaterialFromUserData);
				}
			}
		}

		// get all the inactive entities from the inactive simulation islands
		{
			const hkArray<hkpSimulationIsland*>& inactiveIslands = world.getInactiveSimulationIslands();

			for(int i = 0; i < inactiveIslands.getSize(); i++)
			{
				const hkArray<hkpEntity*>& inactiveEntities = inactiveIslands[i]->getEntities();
				for(int j = 0; j < inactiveEntities.getSize(); j++)
				{
					appendGeometryFromRigidBody(*static_cast<hkpRigidBody*>( inactiveEntities[j] ), geomOut, getMaterialFromUserData);
				}
			}
		}
	}

	// get all the fixed bodies in the world
	if (world.getFixedIsland())
	{
		const hkArray<hkpEntity*>& fixedEntities = world.getFixedIsland()->getEntities();
		for(int j = 0; j < fixedEntities.getSize(); j++)
		{
			appendGeometryFromRigidBody(*static_cast<hkpRigidBody*>( fixedEntities[j] ), geomOut, getMaterialFromUserData);
		}
	}

	world.unmarkForRead();
}

static void _appendWithTransform(hkGeometry& geomInOut, hkGeometry& geomToAppend, const hkTransform& trans, int material)
{
	const int vertexOffset = geomInOut.m_vertices.getSize();

	// Append transformed verts
	hkVector4* verts = geomInOut.m_vertices.expandBy( geomToAppend.m_vertices.getSize() );
	for (int v=0; v  < geomToAppend.m_vertices.getSize(); v++)
	{
		verts[v].setTransformedPos( trans, geomToAppend.m_vertices[v] );
	}

	// Append tris
	hkGeometry::Triangle* tris = geomInOut.m_triangles.expandBy( geomToAppend.m_triangles.getSize() );
	for (int t=0; t < geomToAppend.m_triangles.getSize(); t++)
	{
		hkGeometry::Triangle& tri = tris[t];
		// Copy and reindex
		tri = geomToAppend.m_triangles[t];
		tri.m_a += vertexOffset;
		tri.m_b += vertexOffset;
		tri.m_c += vertexOffset;
		tri.m_material = material;
	}

}

	/// Creates a geometry from a single rigid body
void hkpGeometryConverter::appendGeometryFromRigidBody(const hkpRigidBody& body, hkGeometry& geomInOut, 
																  hkBool getMaterialFromUserData)
{
	const hkpShape* shape = body.getCollidable()->getShape();

	if ( shape == HK_NULL )
		return;

	hkpShapeDisplayBuilder::hkpShapeDisplayBuilderEnvironment env;
	hkpShapeDisplayBuilder shapeBuilder(env);

	hkInplaceArray<hkDisplayGeometry*,8> displayGeometries;
	shapeBuilder.buildDisplayGeometries( shape, displayGeometries );

	for(int i = (displayGeometries.getSize() - 1); i >= 0; i--)
	{
		hkDisplayGeometry* disp = displayGeometries[i];
		disp->buildGeometry();

		if ( disp->getGeometry() == HK_NULL )
		{
			HK_REPORT("Unable to build display geometry from hkpShape geometry data in body " << body.getName() );
			displayGeometries.removeAt(i);
		}
		else
		{
			int material = getMaterialFromUserData ? (int)body.getUserData() : -1;
			// Append it in world space
			hkTransform transform;
			{
				transform.setMul( body.getTransform(), disp->getTransform() );
			}
			_appendWithTransform(geomInOut, *disp->getGeometry(), transform, material);
		}
		disp->removeReference();
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
