/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/PhysicsMigrationUtils/hkpPhysicsMigrationUtils.h>

#include <Common/Serialize/Util/hkRootLevelContainer.h>
#include <Common/Serialize/ResourceDatabase/hkResourceHandle.h>
#include <Common/Serialize/Util/hkSerializeUtil.h>
#include <Common/Base/System/Io/OArchive/hkOArchive.h>
#include <Common/Base/System/Io/IArchive/hkIArchive.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>

// Physics 2012
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/ExtendedMeshShape/hkpExtendedMeshShape.h>
#include <Physics2012/Internal/Collide/BvCompressedMesh/hkpBvCompressedMeshShape.h>
#include <Physics2012/Utilities/Serialize/hkpPhysicsData.h>
#include <Physics2012/Dynamics/Constraint/Breakable/hkpBreakableConstraintData.h>
#include <Physics2012/Internal/Collide/StaticCompound/hkpStaticCompoundShape.h>

// Physics
#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Shape/Convex/Sphere/hknpSphereShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Capsule/hknpCapsuleShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Triangle/hknpTriangleShape.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShapeUtil.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Compressed/hknpCompressedMeshShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Compressed/hknpCompressedMeshShapeCinfo.h>
#include <Physics/Physics/Collide/Shape/Composite/Compound/Dynamic/hknpDynamicCompoundShape.h>
#include <Physics/Physics/Extensions/PhysicsSystem/hknpPhysicsSceneData.h>
#include <Physics/Physics/Collide/Shape/Composite/Compound/hknpShapeInstance.h>
#include <Physics/Physics/Dynamics/World/hknpWorld.h>
#include <Physics/Internal/Dynamics/Solver/ConstraintAtom/hknpConstraintAtomSolver.h>
#include <Physics/Physics/Collide/Shape/Composite/Compound/Static/hknpStaticCompoundShape.h>


namespace
{
	hknpShape* HK_CALL _processShape(
		const hkpShape& shapeToConvert,
		hkArray< hkRefPtr<const hkReferencedObject> >& ownerArray,
		hkPointerMap<const hkpShape*, hknpShape*>* shapesMap )
	{
		// convert the shape
		hknpShape* shape = (shapesMap != HK_NULL) ? shapesMap->getWithDefault( &shapeToConvert, HK_NULL ) : HK_NULL;
		if(!shape)
		{
			shape = hkpPhysicsMigrationUtils::convertShape( shapeToConvert );

			// Conversion failed?
			if(!shape)
			{
				return HK_NULL;
			}

			// Add to dictionary
			if ( shapesMap != HK_NULL )
			{
				shapesMap->insert( &shapeToConvert, shape );
				HK_ON_DEBUG(HK_REPORT("Unique shapes converted: "<<shapesMap->getSize()));
			}

			// Add to owner array
			ownerArray.pushBack( shape );
			shape->removeReference();
		}
		else
		{
			// Allow any owner array to share the shape
			ownerArray.pushBack( shape );
		}

		return shape;
	}

	void HK_CALL _processMaterial( const hkpRigidBody& rigidBody, hknpMaterial& materialOut )
	{
		materialOut.m_dynamicFriction.setReal<true>( rigidBody.getFriction() );
		materialOut.m_staticFriction.setReal<true>( rigidBody.getFriction() );
		materialOut.m_restitution.setReal<true>( rigidBody.getRestitution() );
	}

	void HK_CALL _processDynamicMotionProperties(
		const hkpRigidBody& rigidBody,
		hkVector4Parameter gravity,
		hknpMotionProperties& propsOut )
	{
		// rigidBody.getRigidMotion()->setDeactivationClass(hkpRigidBodyCinfo::SOLVER_DEACTIVATION_OFF);
		hknpMotionProperties dynamicProps;
		dynamicProps.setPreset( hknpMotionPropertiesId::DYNAMIC );
		dynamicProps.m_angularDamping	= rigidBody.getAngularDamping();
		dynamicProps.m_linearDamping	= rigidBody.getLinearDamping();
		dynamicProps.m_maxLinearSpeed	= rigidBody.getMaxLinearVelocity();
		dynamicProps.m_maxAngularSpeed	= rigidBody.getMaxAngularVelocity();
		dynamicProps.m_gravityFactor	= rigidBody.getGravityFactor();
		dynamicProps.setSolverStabilization(
			hknpMotionProperties::SolverStabilizationType(const_cast< hkpRigidBody& >( rigidBody ).getMotion()->getDeactivationClass()), 
			gravity.length<3>().getReal() );
		dynamicProps.setDeactivationParameters();

		
		propsOut = dynamicProps;
	}

	void HK_CALL _processBodyCinfo(
		const hkpRigidBody& rigidBody,
		hknpShape* shape,
		hknpMaterialId materialId,
		hknpMotionId motionId,
		hknpBodyCinfo& bodyCinfoOut )
	{
		bodyCinfoOut.m_shape = shape;
		bodyCinfoOut.m_materialId = materialId;
		bodyCinfoOut.m_motionId = motionId;
		bodyCinfoOut.m_position = rigidBody.getPosition();
		bodyCinfoOut.m_orientation = rigidBody.getRotation();
		bodyCinfoOut.m_collisionFilterInfo = rigidBody.getCollisionFilterInfo();
		bodyCinfoOut.m_name = rigidBody.getName();
		if ( rigidBody.m_localFrame != HK_NULL )
		{
			// Make a ref-counted heap copy
			hkArray<char> buffer;
			hkSerializeUtil::save( rigidBody.m_localFrame.val(), hkLocalFrameClass, hkOArchive( buffer ).getStreamWriter() );
			bodyCinfoOut.m_localFrame.setAndDontIncrementRefCount( hkSerializeUtil::loadObject<hkLocalFrame>( hkIArchive( (void*)buffer.begin(), buffer.getSize() ).getStreamReader() ) );
			HK_ASSERT2( 0x22440191, bodyCinfoOut.m_localFrame != HK_NULL, "Failure to create an hkLocalFrame copy" );
		}
	}

	void HK_CALL _processDynamicMotionCinfo(
		const hkpRigidBody& rigidBody,
		hknpBodyCinfo& bodyCinfo,
		hknpMotionPropertiesId motionPropertiesId,
		hknpMotionCinfo& motionCinfoOut )
	{
		motionCinfoOut.initializeWithMass( &bodyCinfo, 1, rigidBody.getMass() );
		motionCinfoOut.m_motionPropertiesId = motionPropertiesId;
		motionCinfoOut.m_linearVelocity = rigidBody.getLinearVelocity();
		motionCinfoOut.m_angularVelocity = rigidBody.getAngularVelocity();
	}

}	// anonymous namespace


namespace hkpPhysicsMigrationUtils
{
	hknpShape* HK_CALL convertShape( const hkpShape& shapeToConvert )
	{
		hknpConvexShape::BuildConfig configNoShrink;
		configNoShrink.m_shrinkByRadius = false;

		hknpShape::MassConfig massConfig;
		massConfig.m_quality = hknpShape::MassConfig::QUALITY_HIGH;

		hknpShape* shape;
		const hkpShapeContainer* shapeContainer;

		switch ( shapeToConvert.getType() )
		{

		case hkcdShapeType::SPHERE:
			{
				hkpSphereShape* pShape = (hkpSphereShape*)&shapeToConvert;
				hknpSphereShape* newShape = hknpSphereShape::createSphereShape( hkVector4::getZero(), pShape->getRadius());
				shape = newShape;
				break;
			}

		case hkcdShapeType::BOX:
			{
				hkpBoxShape* boxShape = (hkpBoxShape*)&shapeToConvert;
				hkVector4 halfExtents = boxShape->getHalfExtents();
				hkVector4 vertices[8];
				for(int i=0;i<8;++i)
				{
					vertices[i](0)=halfExtents(0)*(i&1?+1:-1);
					vertices[i](1)=halfExtents(1)*(i&2?+1:-1);
					vertices[i](2)=halfExtents(2)*(i&4?+1:-1);
				}
				hknpConvexShape* convexShape = hknpConvexShape::createFromVertices(hkStridedVertices(vertices,8),boxShape->getRadius(), configNoShrink );
				shape = convexShape;
				break;
			}

		case hkcdShapeType::CYLINDER:
			{
				
				hkpCylinderShape* cylinderShape = (hkpCylinderShape*)&shapeToConvert;
				int numVertices = cylinderShape->getNumCollisionSpheres();
				hkInplaceArray<hkVector4,64> vertices; vertices.setSize( numVertices );
				cylinderShape->getCollisionSpheres( (hkSphere*)vertices.begin() );
				hknpConvexShape* convexShape = hknpConvexShape::createFromVertices(vertices, vertices[0](3), configNoShrink);
				shape = convexShape;
				break;
			}

		case hkcdShapeType::CAPSULE:
			{
				hkpCapsuleShape* capsuleShape = (hkpCapsuleShape*)&shapeToConvert;
				hknpConvexShape* convexShape = hknpCapsuleShape::createCapsuleShape(
					capsuleShape->getVertex(0), capsuleShape->getVertex(1), capsuleShape->getRadius() );
				shape = convexShape;
				break;
			}

		case hkcdShapeType::LIST:
			{
				const hkpListShape* listShape = (hkpListShape*)&shapeToConvert;
				int numChildren = listShape->getNumChildShapes();

				hknpDynamicCompoundShape* compoundShape = new hknpDynamicCompoundShape( numChildren );

				int numInstances = 0;
				for( hkpShapeKey key = listShape->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = listShape->getNextKey(key) )
				{
					hkpShapeBuffer buffer;
					const hkpShape* pChildShape = listShape->getChildShape( key, buffer );
					const hknpShape* childShape = convertShape( *pChildShape );
					if ( childShape != HK_NULL )
					{
						hknpShapeInstance instance( childShape, hkTransform::getIdentity() );
						instance.setEnabled( listShape->isChildEnabled(key) );

						compoundShape->addInstances( &instance, 1 );
						childShape->removeReference();
						numInstances++;
					}
				}
				compoundShape->rebuild();

				HK_ASSERT2( 0xf0ffcd23, numInstances != 0, "You cannot convert an empty list shape");
				shape = compoundShape;
				break;
			}

		case hkcdShapeType::CONVEX_VERTICES:
			{
				const hkpConvexVerticesShape* cxShape = (hkpConvexVerticesShape*)&shapeToConvert;
				hkArray<hkVector4> vertices; cxShape->getOriginalVertices(vertices);

				hknpConvexShape* convexShape = hknpConvexShape::createFromVertices( vertices, cxShape->getRadius(), configNoShrink );
				shape = convexShape;
				break;
			}

		case hkcdShapeType::CONVEX_TRANSFORM:
		case hkcdShapeType::CONVEX_TRANSLATE:
			{
				//
				// Convert the child shape and get its transform
				//

				hknpShape* convertedChildShape = HK_NULL;
				hkTransform transform;
				{
					switch( shapeToConvert.getType() )
					{
					case hkcdShapeType::CONVEX_TRANSFORM:
						{
							const hkpConvexTransformShape* ctShape = (hkpConvexTransformShape*)&shapeToConvert;
							convertedChildShape = convertShape( *ctShape->getChildShape() );
							ctShape->getTransform( &transform );
							break;
						}
					case hkcdShapeType::CONVEX_TRANSLATE:
						{
							const hkpConvexTranslateShape* ctShape = (hkpConvexTranslateShape*)&shapeToConvert;
							convertedChildShape = convertShape( *ctShape->getChildShape() );
							transform.set( hkQuaternion::getIdentity(), ctShape->getTranslation() );
							break;
						}
					default:
						break;
					}
				}

				if( convertedChildShape )
				{
					//
					// Bake the transform
					//

					const hknpConvexShape* convexShape = convertedChildShape->asConvexShape();
					if( convexShape )
					{
						// Primitives need special casing
						
						switch( convexShape->getType() )
						{
							case hknpShapeType::SPHERE:
								{
									const hknpSphereShape* ss = static_cast<const hknpSphereShape*>( convexShape );
									hkVector4 center;
									hkVector4Util::transformPoints( transform, &ss->getVertex(0), 1, &center );
									shape = hknpSphereShape::createSphereShape( center, ss->m_convexRadius );
									break;
								}
							case hknpShapeType::CAPSULE:
								{
									const hknpCapsuleShape* cs = static_cast<const hknpCapsuleShape*>( convexShape );
									hkVector4 points[2];
									hkVector4Util::transformPoints( transform, &cs->m_a, 2, points );
									shape = hknpCapsuleShape::createCapsuleShape( points[0], points[1], cs->m_convexRadius );
									break;
								}
							default:
								{
									const int numVerts = hknpConvexShapeUtil::getNumberOfUniqueVertices( convexShape );
									hkArray<hkVector4> vertices( (hkVector4*)convexShape->getVertices(), numVerts, numVerts );
									hknpConvexShape::BuildConfig noShrinkWithTransform = configNoShrink;
									noShrinkWithTransform.m_extraTransform = &transform;
									shape = hknpConvexShape::createFromVertices( vertices, convexShape->m_convexRadius, noShrinkWithTransform );
									break;
								}
						}
					}
					else
					{
						// Horrible fallback
						hkGeometry shapeGeometry;
						convertedChildShape->buildSurfaceGeometry( hknpShape::CONVEX_RADIUS_DISPLAY_NONE, &shapeGeometry );
						hknpConvexShape::BuildConfig noShrinkWithTransform = configNoShrink;
						noShrinkWithTransform.m_extraTransform = &transform;
						shape = hknpConvexShape::createFromVertices( shapeGeometry.m_vertices, convertedChildShape->m_convexRadius, noShrinkWithTransform );
					}
					
					// destroy the temporary shape
					convertedChildShape->removeReference();
				}
				else
				{
					shape = HK_NULL;
				}

				break;
			}
			
		case hkcdShapeType::BV_COMPRESSED_MESH:
			{
				const hkpBvCompressedMeshShape* compressedMeshShape = (const hkpBvCompressedMeshShape*)&shapeToConvert;
				shapeContainer = compressedMeshShape;
				goto PROCESS_SHAPE_COLLECTION;
			}

		case hkcdShapeType::EXTENDED_MESH:
			{
				shapeContainer = static_cast<const hkpExtendedMeshShape*>(&shapeToConvert);
				goto PROCESS_SHAPE_COLLECTION;
			}

		case hkcdShapeType::MOPP:
			{
				hkpMoppBvTreeShape* moppShape =  (hkpMoppBvTreeShape*)&shapeToConvert;
				shapeContainer = moppShape->getShapeCollection();
			}

PROCESS_SHAPE_COLLECTION:;
			{
				hkGeometry geometry;
				hkArray<hknpShapeInstance> childShapes;
				for ( hkpShapeKey key = shapeContainer->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = shapeContainer->getNextKey(key))
				{
					hkpShapeBuffer buffer;
					const hkpShape* childShape = shapeContainer->getChildShape(key, buffer);
					if (childShape->m_type == hkcdShapeType::TRIANGLE )
					{
						const hkpTriangleShape* tShape = (const hkpTriangleShape*)childShape;
						hkGeometry::Triangle& t = geometry.m_triangles.expandOne();
						t.m_a = geometry.m_vertices.getSize(); geometry.m_vertices.pushBack( tShape->getVertex(0));
						t.m_b = geometry.m_vertices.getSize(); geometry.m_vertices.pushBack( tShape->getVertex(1));
						t.m_c = geometry.m_vertices.getSize(); geometry.m_vertices.pushBack( tShape->getVertex(2));
						t.m_material = HKNP_INVALID_SHAPE_TAG;
					}
					else
					{
						hknpShape* convertedChild = convertShape(*childShape);
						if(convertedChild)
						{
							childShapes.expandOne().setShape(convertedChild);
							convertedChild->removeReference();
						}
					}
				}

				if(geometry.m_triangles.getSize())
				{
					hknpDefaultCompressedMeshShapeCinfo meshInfo(&geometry);
					hknpCompressedMeshShape* mesh = new hknpCompressedMeshShape(meshInfo);
					childShapes.expandOne().setShape(mesh);
					mesh->removeReference();
				}

				if(childShapes.getSize() > 1)
				{
					shape = new hknpStaticCompoundShape(childShapes.begin(), childShapes.getSize());
				}
				else if(childShapes.getSize() == 1)
				{
					shape = const_cast<hknpShape*>(childShapes[0].getShape());
					shape->addReference();
				}
				else
				{
					shape = HK_NULL;
				}
				break;
			}

		case hkcdShapeType::TRIANGLE:
			{
				hkpTriangleShape* tShape = (hkpTriangleShape*)&shapeToConvert;
				hknpTriangleShape* triangleShape = hknpTriangleShape::createTriangleShape(
					tShape->getVertex(0), tShape->getVertex(1), tShape->getVertex(2), tShape->getRadius() );
				shape = triangleShape;
				break;
			}

		case hkcdShapeType::STATIC_COMPOUND:
			{
				hkArray<hknpShapeInstance>::Temp instances;
				const hkpStaticCompoundShape* origSCS = static_cast<const hkpStaticCompoundShape*>(&shapeToConvert);
				for(int i = 0; i < origSCS->getInstances().getSize(); i++)
				{
					const hkQsTransform& qt = origSCS->getInstances()[i].getTransform();
					hkTransform t; t.setRotation(qt.getRotation()); t.setTranslation(qt.getTranslation());
					hknpShapeInstance instance(convertShape(*origSCS->getInstances()[i].getShape()), t);
					instance.setScale(qt.getScale());
					instances.pushBack(instance);
				}
				hknpStaticCompoundShape* scs = new hknpStaticCompoundShape(instances.begin(), instances.getSize());
				shape = scs;
				break;
			}

		default:
			{
				HK_WARN_ALWAYS(0x2d1f57c2, "Shape type(" << shapeToConvert.getType() << ") is not supported for conversion.");
				return HK_NULL;
			}

		}

		if ( shape != HK_NULL ) shape->setMassProperties( massConfig );

		return shape;
	}

	hknpBodyCinfo* HK_CALL convertBody(
		const hkpRigidBody& body,
		hkVector4Parameter gravity,
		hknpPhysicsSystemData& systemDataOut,
		hkPointerMap<const hkpShape*, hknpShape*>* shapesMap )
	{
		// Convert the shape
		if ( body.getCollidable()->getShape() == HK_NULL )
		{
			return HK_NULL;
		}
		hknpShape* shape = _processShape( *body.getCollidable()->getShape(), systemDataOut.m_referencedObjects, shapesMap );
		if ( shape == HK_NULL )
		{
			return HK_NULL;
		}

		// Convert the material
		hknpMaterialId materialId;
		{
			materialId = hknpMaterialId( systemDataOut.m_materials.getSize() );
			hknpMaterial& material = systemDataOut.m_materials.expandOne();
			_processMaterial( body, material );
		}

		// Convert the motion properties
		hknpMotionPropertiesId motionPropertiesId;
		{
			if ( body.getMotionType() == hkpMotion::MOTION_KEYFRAMED )
			{
				// Invalid motionPropertiesIds get set to hknpMotionPropertiesId::KEYFRAMED in the hknpPhysicsSystem ctor.
				motionPropertiesId = hknpMotionPropertiesId::invalid();
			}
			else if ( !body.isFixedOrKeyframed() )
			{
				// Create a dynamic motionPropertiesId
				motionPropertiesId = hknpMotionPropertiesId( systemDataOut.m_motionProperties.getSize() );
				hknpMotionProperties& motionProperties = systemDataOut.m_motionProperties.expandOne();
				_processDynamicMotionProperties( body, gravity, motionProperties );
			}

			// Fixed bodies will never use a motionPropertiesId, they will be assigned STATIC motion in hknpPhysicsSystem ctor.
		}

		// Convert the motion
		hknpMotionId motionId;
		hknpMotionCinfo* motionCinfo = HK_NULL;
		if( body.isFixed() )
		{
			// Invalid motionIds get set to hknpMotionId::STATIC in the hknpPhysicsSystem ctor.
			motionId = hknpMotionId::invalid();
		}
		else
		{
			motionId = hknpMotionId( systemDataOut.m_motionCinfos.getSize() );
			motionCinfo = &systemDataOut.m_motionCinfos.expandOne();
		}

		// Put it all together
		hknpBodyCinfo& bodyCinfo = systemDataOut.m_bodyCinfos.expandOne();
		_processBodyCinfo( body, shape, materialId, motionId, bodyCinfo );
		
		if ( body.getMotionType() == hkpMotion::MOTION_KEYFRAMED )
		{
			motionCinfo->initializeAsKeyFramed( &bodyCinfo, 1 );
			motionCinfo->m_orientation = body.getRotation();
			motionCinfo->m_motionPropertiesId = motionPropertiesId;
			motionCinfo->m_linearVelocity = body.getLinearVelocity();
			motionCinfo->m_angularVelocity = body.getAngularVelocity();
		}
		else if ( !body.isFixedOrKeyframed() )
		{
			_processDynamicMotionCinfo( body, bodyCinfo, motionPropertiesId, *motionCinfo );
		}

		return &bodyCinfo;
	}

	void HK_CALL convertConstraint(
		const hkpConstraintInstance& constraint,
		hknpConstraintCinfo& constraintCinfoOut,
		const hkArrayBase<hkpRigidBody*>& bodyMap )
	{
		hkpConstraintData* data = constraint.getDataRw();

		if ( data->getType() == hkpConstraintData::CONSTRAINT_TYPE_CONTACT )
		{
			return;
		}

		
		if ( data->getType() == hkpConstraintData::CONSTRAINT_TYPE_BREAKABLE )
		{
			HK_ASSERT2(0x22440106, false, "Breakable constraints are not currently supported.");
			data = static_cast<hkpBreakableConstraintData*>( data )->getWrappedConstraintData();
		}

		hkpRigidBody* bodyA = constraint.getRigidBodyA();
		hkpRigidBody* bodyB = constraint.getRigidBodyB();

		int bodyAIdx = bodyMap.indexOf( bodyA );

		constraintCinfoOut.m_constraintData = data;
		constraintCinfoOut.m_bodyA = hknpBodyId(bodyAIdx);
		if(bodyB)
		{
			int bodyBIdx = bodyMap.indexOf( bodyB );
		constraintCinfoOut.m_bodyB = hknpBodyId(bodyBIdx);
		}
		else
		{
			constraintCinfoOut.m_bodyB = hknpBodyId::INVALID;
		}
	}
	
	void HK_CALL convertPhysicsSystem(
		const hkpPhysicsSystem& physicsSystem,
		hknpPhysicsSystemData& systemDataOut,
		const hkVector4* newWorldGravity /*= HK_NULL */,
		hkPointerMap<const hkpShape*, hknpShape*>* shapesMap /*= HK_NULL */ )
	{
		bool internalShapeMape = false;
		if (shapesMap == HK_NULL)
		{
			shapesMap = new hkPointerMap< const hkpShape*, hknpShape*>();
			internalShapeMape = true;
		}
		
		// Copy name
		systemDataOut.m_name = physicsSystem.getName();

		// Convert bodies
		const hkArray<hkpRigidBody*>& rigidBodies = physicsSystem.getRigidBodies();
		{
			for( int rbIter = 0; rbIter < rigidBodies.getSize(); rbIter++ )
			{
				// Gravity is only used to set a solver stabilization factor...only it's magnitude matters.
				
				hkVector4 gravity;
				gravity.set( 0.0f, 0.0f, -9.8f, 0.0f );
				hkpPhysicsMigrationUtils::convertBody( *rigidBodies[rbIter], (newWorldGravity != HK_NULL) ? *newWorldGravity : gravity, systemDataOut, shapesMap );
			}
		}

		// Convert constraints
		{
			const hkArray<hkpConstraintInstance*>& constraints = physicsSystem.getConstraints();
			for( int cIter = 0; cIter < constraints.getSize(); cIter++ )
			{
				hknpConstraintCinfo& constraintCinfo = systemDataOut.m_constraintCinfos.expandOne();
				convertConstraint( *constraints[cIter], constraintCinfo, rigidBodies );
			}
		}

		HK_ASSERT2(0x65c12cd1, physicsSystem.getActions().getSize() == 0, "Actions are currently not converted.");
		HK_ASSERT2(0x65c12cd1, physicsSystem.getPhantoms().getSize() == 0, "Phantoms are currently not converted.");
		HK_ASSERT2(0x65c12cd1, physicsSystem.getUserData() == HK_NULL, "User data is currently not converted.");
		
		if (internalShapeMape)
		{
			delete shapesMap;
		}
	}

	bool HK_CALL convertRootLevelContainer(
		const hkRootLevelContainer& rootLevelContainer,
		hkRootLevelContainer& rootLevelContainerOut,
		const hkVector4* newWorldGravity,
		hkPointerMap<const hkpShape*, hknpShape*>* shapesMap,
		bool pruneNonConvertedPhysicsData )
	{
		// Note if we make a change of any kind
		bool changed = false;

		bool internalShapeMape = false;
		if (shapesMap == HK_NULL)
		{
			shapesMap = new hkPointerMap< const hkpShape*, hknpShape*>();
			internalShapeMape = true;
		}

		// Iterate over old data, convert as we go
		for( int i = 0; i < rootLevelContainer.m_namedVariants.getSize(); ++i )
		{
			if( rootLevelContainer.m_namedVariants[i].getTypeName() && hkString::strCmp( hkpPhysicsDataClass.getName(), rootLevelContainer.m_namedVariants[i].getTypeName() ) == 0 )
			{
				// Grab old data info
				hkpPhysicsData* physicsData = reinterpret_cast<hkpPhysicsData*>( rootLevelContainer.m_namedVariants[i].getObject() );
				hkStringPtr physicsDataName;
				if ( &rootLevelContainer == &rootLevelContainerOut )
				{
					physicsDataName.printf( "_%s_converted_np", rootLevelContainer.m_namedVariants[i].getName() );
				}
				else
				{
					physicsDataName = rootLevelContainer.m_namedVariants[i].getName();
				}

				// Convert it
				
				
				{
					hknpPhysicsSceneData* npPhysicsData = new hknpPhysicsSceneData();

					for ( int systemIter = 0; systemIter < physicsData->getPhysicsSystems().getSize(); systemIter++ )
					{
						const hkpPhysicsSystem* physicsSystem = physicsData->getPhysicsSystems()[systemIter];

						hknpPhysicsSystemData* npPhysicsSystem = new hknpPhysicsSystemData();
						convertPhysicsSystem( *physicsSystem, *npPhysicsSystem, newWorldGravity, shapesMap );

						npPhysicsData->m_systemDatas.expandOne().setAndDontIncrementRefCount( npPhysicsSystem );
					}

					hkRootLevelContainer::NamedVariant& namedVariant = rootLevelContainerOut.m_namedVariants.expandOne();
					namedVariant.set( physicsDataName, npPhysicsData, &hknpPhysicsSceneDataClass );
					npPhysicsData->removeReference();
				}

				changed = true;
			}
			else if ( &rootLevelContainer != &rootLevelContainerOut )
			{
				// Pass object through to new container
				
				
				
				{
					rootLevelContainerOut.m_namedVariants.expandOne() = rootLevelContainer.m_namedVariants[i];
				}
			}
		}

		// Prune any physics data if requested
		if ( pruneNonConvertedPhysicsData )
		{
			changed |= pruneDeprecatedClasses( rootLevelContainerOut );
		}

		if (internalShapeMape)
		{
			delete shapesMap;
		}

		return changed;
	}

	hknpBodyId HK_CALL addBody(
		hknpWorld& world,
		const hkpRigidBody& body,
		hkPointerMap<const hkpShape*, hknpShape*>* shapesMap,
		bool forceKeyframed )
	{
		// Convert the shape
		if ( body.getCollidable()->getShape() == HK_NULL )
		{
			return hknpBodyId::invalid();
		}
		hknpShape* shape = _processShape( *body.getCollidable()->getShape(), world.m_userReferencedObjects, shapesMap );
		if ( shape == HK_NULL )
		{
			return hknpBodyId::invalid();
		}

		// Convert the material
		hknpMaterialId materialId;
		{
			hknpMaterial material;
			_processMaterial( body, material );
			materialId = world.accessMaterialLibrary()->addEntry( material );
		}

		// Convert the motion properties
		hknpMotionPropertiesId motionPropertiesId;
		{
			if (body.getMotionType() == hkpMotion::MOTION_KEYFRAMED)
			{
				motionPropertiesId = hknpMotionPropertiesId::KEYFRAMED;
			}
			else if ( !body.isFixedOrKeyframed() )
			{
				hknpMotionProperties motionProperties;
				_processDynamicMotionProperties( body, world.getGravity(), motionProperties );
				motionPropertiesId = world.accessMotionPropertiesLibrary()->addEntry( motionProperties );
			}

			// Fixed bodies will never use a motionPropertiesId.
		}

		// Allow the world to assign the motion (creating a dynamic one if necessary)
		hknpMotionId motionId = hknpMotionId::invalid();

		// Put it all together
		hknpBodyCinfo bodyCinfo;
		_processBodyCinfo( body, shape, materialId, motionId, bodyCinfo );

		// Create/Add new body
		if ( body.isFixed() )
		{
			return world.createStaticBody( bodyCinfo, hknpWorld::ADD_BODY_NOW );
		}
		else
		{
			hknpMotionCinfo motionCinfo;
			if ( forceKeyframed || ( body.getMotionType() == hkpMotion::MOTION_KEYFRAMED ) )
			{
				motionCinfo.initializeAsKeyFramed( &bodyCinfo, 1 );
				motionCinfo.m_orientation = body.getRotation();
				motionCinfo.m_motionPropertiesId = hknpMotionPropertiesId::KEYFRAMED;
				motionCinfo.m_linearVelocity = body.getLinearVelocity();
				motionCinfo.m_angularVelocity = body.getAngularVelocity();
			}
			else
			{
				_processDynamicMotionCinfo( body, bodyCinfo, motionPropertiesId, motionCinfo );
			}

			return world.createDynamicBody( bodyCinfo, motionCinfo, hknpWorld::ADD_BODY_NOW );
		}
	}

	void HK_CALL removeBody(
		hknpWorld& world,
		const hkpRigidBody& body,
		hkPointerMap<const hkpShape*, hknpShape*>* shapesMap )
	{
		const hkpShape* oldShape = body.getCollidable()->getShape();
		if ( shapesMap && shapesMap->hasKey( oldShape ) )
		{
			shapesMap->remove( oldShape );
		}

		hknpBodyId rbId = hknpBodyId(body.m_npData);
		if ( !rbId.isValid() )
		{
			return;
		}
		const hknpBody& npBody = world.getBody(rbId);
		if ( npBody.m_shape )
		{
			npBody.m_shape->removeReference();
		}
		world.destroyBodies( &rbId, 1 );
	}

	bool _hasSuperClass(const hkClass* klass, const char* superClassName)
	{
		if (hkString::strCmp(klass->getName(), superClassName) == 0)
		{
			return true;
		}
		const hkClass* parentClass = klass->getParent();
		if (parentClass)
		{
			return _hasSuperClass(parentClass, superClassName);
		}
		return false;
	}

	hkRootLevelContainer::NamedVariant HK_CALL pruneDeprecatedDestructionClasses( const hkRootLevelContainer::NamedVariant& namedVariant, bool& changed )
	{
		
		if ( hkString::strCmp(namedVariant.getName(), "Resource Data") == 0 )
		{
			hkResourceContainer* resource_Data = reinterpret_cast<hkResourceContainer*>(namedVariant.getObject());
			hkArray<hkResourceContainer*> containers;
			resource_Data->findAllContainersRecursively(containers);

			for (int i = 0; i < containers.getSize(); i++)
			{
				hkResourceContainer* container = containers[i];

				hkArray<hkResourceHandle*> handlesToDelete;
				for (hkResourceHandle* handle = container->findResourceByName(HK_NULL, HK_NULL, HK_NULL); handle;  handle = container->findResourceByName(HK_NULL, HK_NULL, handle))
				{
					const hkClass* klass = handle->getClass();

					if (isDeprecatedClass(klass->getName()))
					{
						handlesToDelete.pushBack(handle);
					}
				}

				for (int handleIndex = 0; handleIndex < handlesToDelete.getSize(); handleIndex++)
				{
					container->destroyResource(handlesToDelete[handleIndex]);
					changed = true;
				}

				int numHandlesLeft = container->getNumResources();
				if ((numHandlesLeft == 0) && (container->getParent() != HK_NULL))
				{
					container->getParent()->destroyContainer(container);
					changed = true;
				}
			}
		}
		
		return namedVariant;
	}

	bool HK_CALL pruneDeprecatedClasses(
		hkRootLevelContainer& rootLevelContainer )
	{
		// monitor if we've made changes
		bool changed = false;

		// Iterate over old data, prune as we go
		for( int i = rootLevelContainer.m_namedVariants.getSize()-1; i >= 0 ; --i )
		{
			// Reference non-physics data in the new container if it's not the same as our source.

			const char* typeName = rootLevelContainer.m_namedVariants[i].getTypeName();
			if ( typeName != HK_NULL )
			{
				// Is the class directly deprecated
				if ( isDeprecatedClass( typeName ) )
				{
					changed = true;
					rootLevelContainer.m_namedVariants.removeAt(i);
					continue;
				}
			}

			// Pass object through to new container
			
			
			
			{
				rootLevelContainer.m_namedVariants[i] = pruneDeprecatedDestructionClasses( rootLevelContainer.m_namedVariants[i], changed );
			}
		}

		return changed;
	}

#define HK_CLASS(str) #str,
#define HK_ABSTRACT_CLASS(str) #str,
#define HK_STRUCT(str) #str,
	static const char* EXCEPTION_CLASS_NAMES[] = {
#include <Physics/Constraint/Classes/hkpConstraintClasses.h>
#include <Physics/ConstraintSolver/Classes/hkpConstraintSolverClasses.h>
	};
#undef HK_CLASS
#undef HK_ABSTRACT_CLASS
#undef HK_STRUCT

	bool HK_CALL isDeprecatedClass( const char* className )
	{
		bool hkdClass = hkString::beginsWith( className, "hkd");
		if (hkdClass)
		{
			return true;
		}

		bool hkpClass = hkString::beginsWith( className, "hkp" );
		if (!hkpClass)
		{
			return false;
		}

		const int exceptionListSize = sizeof(EXCEPTION_CLASS_NAMES) / sizeof(char*);
		for( int classNameIt = 0; classNameIt < exceptionListSize; classNameIt++ )
		{
			const char* exceptionClass = EXCEPTION_CLASS_NAMES[classNameIt];

			if ( hkString::strCmp( className, exceptionClass ) == 0 )
			{
				return false;
			}
		}

		return true;
	}

} // hkpPhysicsMigrationUtils namespace

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
