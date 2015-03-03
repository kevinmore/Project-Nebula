/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpPhysicsSystem.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>

#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>
#include <Physics2012/Collide/Util/Welding/hkpMeshWeldingUtility.h>

#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/Shape/hkDisplayConvex.h>

#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpInconsistentWindingViewer.h>

int hkpInconsistentWindingViewer::m_tag = 0;


void HK_CALL hkpInconsistentWindingViewer::registerViewer( void )
{
	m_tag = hkProcessFactory::getInstance().registerProcess( getName(), create );
}

hkProcess* HK_CALL hkpInconsistentWindingViewer::create( const hkArray<hkProcessContext*>& contexts )
{
	return new hkpInconsistentWindingViewer( contexts );
}

hkpInconsistentWindingViewer::hkpInconsistentWindingViewer(const hkArray<hkProcessContext*>& contexts)
: hkpWorldViewerBase( contexts )
{
}

void hkpInconsistentWindingViewer::init()
{
	if ( m_context )
	{
		for ( int k = 0; k < m_context->getNumWorlds(); ++k )
		{
			hkpWorld* world = m_context->getWorld( k );

			world->markForWrite();
			world->addWorldPostSimulationListener( this );
			world->addEntityListener( this );
			// get all the inactive entities from the inactive simulation islands
			{
				const hkArray<hkpSimulationIsland*>& inactiveIslands = world->getInactiveSimulationIslands();

				for(int i = 0; i < inactiveIslands.getSize(); i++)
				{
					const hkArray<hkpEntity*>& activeEntities = inactiveIslands[i]->getEntities();
					for(int j = 0; j < activeEntities.getSize(); j++)
					{
						entityAddedCallback( activeEntities[j] );
					}
				}
			}


			// get all the fixed bodies in the world
			if (world->getFixedIsland())
			{
				const hkArray<hkpEntity*>& fixedEntities = world->getFixedIsland()->getEntities();
				for(int j = 0; j < fixedEntities.getSize(); j++)
				{
					entityAddedCallback( fixedEntities[j] );
				}
			}
			world->unmarkForWrite();
		}
	}
}

hkpInconsistentWindingViewer::~hkpInconsistentWindingViewer()
{
	if ( m_context )
	{
		for ( int i = 0; i < m_context->getNumWorlds(); ++i )
		{
			hkpWorld* world = m_context->getWorld( i );

			world->markForWrite();
			world->removeWorldPostSimulationListener( this );
			world->removeEntityListener( this );
			world->unmarkForWrite();
		}
	}
}

void hkpInconsistentWindingViewer::worldAddedCallback( hkpWorld* world )
{
	world->markForWrite();;
	world->addWorldPostSimulationListener(this);
	world->addEntityListener( this );
	// get all the inactive entities from the inactive simulation islands
	{
		const hkArray<hkpSimulationIsland*>& inactiveIslands = world->getInactiveSimulationIslands();

		for(int i = 0; i < inactiveIslands.getSize(); i++)
		{
			const hkArray<hkpEntity*>& activeEntities = inactiveIslands[i]->getEntities();
			for(int j = 0; j < activeEntities.getSize(); j++)
			{
				entityAddedCallback( activeEntities[j] );
			}
		}
	}


	// get all the fixed bodies in the world
	if (world->getFixedIsland())
	{
		const hkArray<hkpEntity*>& fixedEntities = world->getFixedIsland()->getEntities();
		for(int j = 0; j < fixedEntities.getSize(); j++)
		{
			entityAddedCallback( fixedEntities[j] );
		}
	}
	world->unmarkForWrite();
}

void hkpInconsistentWindingViewer::worldRemovedCallback( hkpWorld* world )
{
	world->markForWrite();;
	world->removeWorldPostSimulationListener(this);
	world->removeEntityListener( this );
	world->unmarkForWrite();
}

void hkpInconsistentWindingViewer::entityAddedCallback( hkpEntity* entity )
{
	
	hkpRigidBody* rigidBody = static_cast<hkpRigidBody*>(entity);

	const hkpShape* shape = rigidBody->getCollidable()->getShape();

	if (shape == HK_NULL)
	{
		return; // nothing to build from
	}

	if ( shape->getType() == hkcdShapeType::MOPP )
	{
		// ... and process all triangles if the shape is a MOPP..
		const hkpMoppBvTreeShape* mopp = static_cast< const hkpMoppBvTreeShape* >( shape );
		hkGeometry* geom = new hkGeometry;
		const hkTransform& bodyTransform = rigidBody->getTransform();
		const hkpShapeCollection* coll = mopp->getShapeCollection();
		hkpShapeBuffer buffer;
		for ( hkpShapeKey key = coll->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = coll->getNextKey( key ) )
		{				
			const hkpShape* childShape = coll->getChildShape( key, buffer );

			if ( childShape->getType() == hkcdShapeType::TRIANGLE )
			{
				const hkpTriangleShape* triangle = static_cast< const hkpTriangleShape* >( childShape );

				/// ... get the welding info for the triangle...
				if ( triangle->getWeldingType() != hkpWeldingUtility::WELDING_TYPE_TWO_SIDED && triangle->getWeldingType() != hkpWeldingUtility::WELDING_TYPE_NONE )
				{
					hkUint16 weldingInfo = triangle->getWeldingInfo();

					if ( ( weldingInfo >> 15 ) != 0 )
					{
						hkVector4 vertices[ 3 ];
						vertices[ 0 ]._setTransformedPos( bodyTransform, triangle->getVertex<0>() );
						vertices[ 1 ]._setTransformedPos( bodyTransform, triangle->getVertex<1>() );
						vertices[ 2 ]._setTransformedPos( bodyTransform, triangle->getVertex<2>() );

						hkGeometry::Triangle geomTriangle;
						geomTriangle.m_a = geom->m_vertices.getSize();
						geomTriangle.m_b = geomTriangle.m_a + 1;
						geomTriangle.m_c = geomTriangle.m_a + 2;
						geomTriangle.m_material = 0;
						geom->m_vertices.pushBack( vertices[0] );
						geom->m_vertices.pushBack( vertices[1] );
						geom->m_vertices.pushBack( vertices[2] );
						geom->m_triangles.pushBack( geomTriangle );
					}			
				}
			}
		}

		// add the shape to the display handler
		if (geom->m_vertices.getSize() != 0 && geom->m_triangles.getSize() != 0)
		{
			hkArray<hkDisplayGeometry*> displayGeometries;
			hkDisplayGeometry* displayGeom = new hkDisplayConvex( geom );
			displayGeometries.pushBack( displayGeom );
			hkUlong id = (hkUlong)coll;
			m_displayHandler->addGeometry( displayGeometries, hkTransform::getIdentity(), id, m_tag, (hkUlong)shape );
			m_displayHandler->setGeometryColor( hkColor::GREEN, id, m_tag );
		}
		else
		{
			delete geom;
		}
	}
}

void hkpInconsistentWindingViewer::postSimulationCallback( hkpWorld* world )
{
	HK_TIME_CODE_BLOCK("hkpInconsistentWindingViewer::postSimulationCallback", HK_NULL);

	world->lock();

	// For all rigid bodies...


	hkpPhysicsSystem* system = world->getWorldAsOneSystem();
	const hkArray< hkpRigidBody* >& rigidBodies = system->getRigidBodies();
	for ( int i = 0; i < rigidBodies.getSize(); ++i )
	{
		const hkpRigidBody* body = rigidBodies[ i ];
		const int collidableId = (int)(hkUlong)(body->getCollidable());
		const hkTransform& bodyTransform = body->getTransform();
		const hkpShape* rootShape = body->getCollidable()->getShape();

		if ( rootShape->getType() == hkcdShapeType::MOPP )
		{
			// ... and process all triangles if the shape is a MOPP..
			const hkpMoppBvTreeShape* mopp = static_cast< const hkpMoppBvTreeShape* >( rootShape );

			const hkpShapeCollection* coll = mopp->getShapeCollection();
			for ( hkpShapeKey key = coll->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = coll->getNextKey( key ) )
			{
				hkpShapeBuffer buffer;
				const hkpShape* childShape = coll->getChildShape( key, buffer );
				
				if ( childShape->getType() == hkcdShapeType::TRIANGLE )
				{
					const hkpTriangleShape* triangle = static_cast< const hkpTriangleShape* >( childShape );

					/// ... get the welding info for the triangle...
					if ( triangle->getWeldingType() != hkpWeldingUtility::WELDING_TYPE_TWO_SIDED && triangle->getWeldingType() != hkpWeldingUtility::WELDING_TYPE_NONE )
					{
						hkUint16 weldingInfo = triangle->getWeldingInfo();

						if ( ( weldingInfo >> 15 ) == 0 )
						{
							continue;
						}
						
						hkVector4 vertices[ 3 ];
						vertices[ 0 ]._setTransformedPos( bodyTransform, triangle->getVertex<0>() );
						vertices[ 1 ]._setTransformedPos( bodyTransform, triangle->getVertex<1>() );
						vertices[ 2 ]._setTransformedPos( bodyTransform, triangle->getVertex<2>() );

						hkVector4 triangleCenter; 
						triangleCenter.setAdd( vertices[0], vertices[1] );
						triangleCenter.add( vertices[2] );
						triangleCenter.mul(hkSimdReal_Inv3);

						hkVector4 triangleNormal;
						hkpMeshWeldingUtility::calcAntiClockwiseTriangleNormal( vertices[ 0 ], vertices[ 1 ], vertices[ 2 ], triangleNormal );
						
						m_displayHandler->displayArrow( triangleCenter, triangleNormal, hkColor::ORANGE, collidableId, m_tag );

						// ...and draw the snap vectors for each edge 
						for ( int k = 0; k < 3; ++k )
						{
							int edgeIndex0 = k;
							int edgeIndex1 = ( k + 1 ) % 3;

							hkVector4 edge;
							edge.setSub( vertices[ edgeIndex1 ], vertices[ edgeIndex0 ] );
							edge.normalize<3>();

							int edgeCode = hkpMeshWeldingUtility::calcSingleEdgeBitcode( weldingInfo, k );
							if ( edgeCode == hkpWeldingUtility::NUM_ANGLES )
							{
								m_displayHandler->displayLine( vertices[ edgeIndex0 ], vertices[ edgeIndex1 ], hkColor::RED, collidableId, m_tag );
							}
							else
							{
								m_displayHandler->displayLine( vertices[ edgeIndex0 ], vertices[ edgeIndex1 ], hkColor::BLUE, collidableId, m_tag );
							}
						}							
					}
				}
			}
		}
	}

	system->removeReference();
	world->unlock();
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
