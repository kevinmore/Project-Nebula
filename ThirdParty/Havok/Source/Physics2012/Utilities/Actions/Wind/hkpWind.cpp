/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Actions/Wind/hkpWind.h>

#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesConnectivityUtil.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>

#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplay.h>

#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTransform/hkpConvexTransformShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Deprecated/ConvexList/hkpConvexListShape.h>
#include <Physics2012/Collide/Shape/Misc/Bv/hkpBvShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>
#include <Physics2012/Collide/Shape/hkpShapeContainer.h>

// ////////////////////////////////////////////////////////////////////////
// WindOnShape implementation
// ////////////////////////////////////////////////////////////////////////

/// The class which implements the algorithm for applying wind to a shape.


class hkpWind::WindOnShape
{
	public:
			/// Constructor for wind only.
		WindOnShape( hkpRigidBody* rb, hkReal deltaTime, hkReal obbFactor, const hkpWind* wind );
			/// Constructor for resistance only.
		WindOnShape( hkpRigidBody* rb, hkReal deltaTime, hkReal obbFactor, hkReal resistanceFactor );
			/// Constructor for wind and resistance.
		WindOnShape( hkpRigidBody* rb, hkReal deltaTime, hkReal obbFactor, const hkpWind* wind, hkReal resistanceFactor );

			/// Apply the wind forces acting on an unknown shape.
		void applyWindToShape();
	private:
			/// The accumulated force.
		hkVector4 m_force;
			/// The accumulated torque.
		hkVector4 m_torque;
			/// The body to which wind is applying.
		hkpRigidBody* m_body;
			/// The length of this time step.
		hkReal m_deltaTime;
			/// The OBB factor.
		hkReal m_obbFactor;
			/// A pointer to the wind object or HK_NULL.
		const hkpWind* m_wind;
			/// The resistance factor (which may be 0.0f).
		hkReal m_resistanceFactor;

	private:
			// Obtain the wind at a given point.
		void getWindVector( const hkVector4& pos, hkVector4& windOut ) const;

			// Combine a force at position worldPoint into the accumulated force and torque.
		void accumulate( const hkVector4& worldPoint, const hkVector4& force );

	private:
			/// Apply the wind forces acting on an unknown shape.
		void accumulateForcesOnShape( const hkpShape& shape, const hkTransform& transformToWorld );

			/// Apply the wind forces acting on a sphere.
		void accumulateForcesOnSphere( const hkpSphereShape& sphere, const hkTransform& transformToWorld );

			/// Apply the wind forces acting on a box.
		void accumulateForcesOnBox( const hkpBoxShape& box, const hkTransform& transformToWorld );

			/// Apply the wind forces acting on a triangle.
		void accumulateForcesOnTriangle( const hkpTriangleShape& triangle, const hkTransform& transformToWorld );

			/// Common calculation of the wind forces acting on the side of a cylinder or capsule.
		void accumulateForcesOnCylinderSide( const hkVector4& worldA, const hkVector4& worldB, hkReal radius, hkVector4& normalAtAOut, hkVector4& windVectorOut );

			/// Apply the wind forces acting on a cylinder.
		void accumulateForcesOnCylinder( const hkpCylinderShape& cylinder, const hkTransform& transformToWorld );

			/// Apply the wind forces acting on a capsule.
		void accumulateForcesOnCapsule( const hkpCapsuleShape& capsule, const hkTransform& transformToWorld );		

			/// Apply the wind forces acting on a convex vertices shape.
		void accumulateForcesOnConvexVertices( const hkpConvexVerticesShape& cvShape, const hkTransform& transformToWorld );

			/// A fall-back implementation which uses oriented-bounding boxes to determine wind force.
		void accumulateForcesOnObb( const hkpShape& sphape, const hkTransform& transformToWorld );
};

// ////////////////////////////////////////////////////////////////////////
// hkpWind implementation
// ////////////////////////////////////////////////////////////////////////


void hkpWind::applyWind( hkpRigidBody* rb, hkReal deltaTime, hkReal obbFactor ) const
{
	WindOnShape callback( rb, deltaTime, obbFactor, this );

	callback.applyWindToShape();
}

void hkpWind::applyWindAndResistance( hkpRigidBody* rb, hkReal deltaTime, hkReal resistanceFactor, hkReal obbFactor ) const
{
	// Combine the air properties with relative motion via this callback object.
	WindOnShape callback( rb, deltaTime, obbFactor, this, resistanceFactor );

	callback.applyWindToShape();
}

void hkpWind::applyResistance( hkpRigidBody* rb, hkReal deltaTime, hkReal resistanceFactor, hkReal obbFactor )
{
	WindOnShape callback( rb, deltaTime, obbFactor, resistanceFactor );

	callback.applyWindToShape();
}

// ////////////////////////////////////////////////////////////////////////
// WindOnShape implementation
// ////////////////////////////////////////////////////////////////////////

hkpWind::WindOnShape::WindOnShape( hkpRigidBody* rb, hkReal deltaTime, hkReal obbFactor, const hkpWind* wind, hkReal resistanceFactor )
:	m_body( rb ),
	m_deltaTime( deltaTime ),
	m_obbFactor( obbFactor ),
	m_wind( wind ),
	m_resistanceFactor( resistanceFactor )
{
	m_force.setZero();
	m_torque.setZero();
}

hkpWind::WindOnShape::WindOnShape( hkpRigidBody* rb, hkReal deltaTime, hkReal obbFactor, const hkpWind* wind )
:	m_body( rb ),
	m_deltaTime( deltaTime ),
	m_obbFactor( obbFactor ),
	m_wind( wind ),
	m_resistanceFactor( 0.0f )
{
	m_force.setZero();
	m_torque.setZero();
}

hkpWind::WindOnShape::WindOnShape( hkpRigidBody* rb, hkReal deltaTime, hkReal obbFactor, hkReal resistanceFactor )
:	m_body( rb ),
	m_deltaTime( deltaTime ),
	m_obbFactor( obbFactor ),
	m_wind( HK_NULL ),
	m_resistanceFactor( resistanceFactor )
{
	m_force.setZero();
	m_torque.setZero();
}

void hkpWind::WindOnShape::applyWindToShape()
{
	hkpRigidBodyCinfo info;
	m_body->getCinfo(info);
	accumulateForcesOnShape( *info.m_shape, m_body->getTransform() );

	m_body->applyForce( m_deltaTime, m_force );
	m_body->applyTorque( m_deltaTime, m_torque );

	//HK_DISPLAY_ARROW( m_body->getCenterOfMassInWorld(), m_force, hkColor::BLUE);
}

inline void hkpWind::WindOnShape::accumulate(const hkVector4 &worldPoint, const hkVector4 &force)
{
	hkVector4 pointRel;
	{
		pointRel.setSub( worldPoint, m_body->getCenterOfMassInWorld() );
	}

	hkVector4 torque;
	{
		torque.setCross( pointRel, force );
	}

	m_force.setAdd( m_force, force );
	m_torque.setAdd( m_torque, torque );

	//HK_DISPLAY_ARROW( worldPoint, force, hkColor::GREEN);
}

inline void hkpWind::WindOnShape::getWindVector( const hkVector4& pos, hkVector4& windOut ) const
{
	if ( m_wind )
	{
		m_wind->getWindVector( pos, windOut );
	}
	else
	{
		windOut.setZero();
	}
	const hkSimdReal resist = hkSimdReal::fromFloat(m_resistanceFactor);
	if ( resist.isGreaterZero() )
	{
		hkVector4 relWind; m_body->getPointVelocity( pos, relWind );
		windOut.addMul( -resist, relWind );
	}
}

void hkpWind::WindOnShape::accumulateForcesOnShape( const hkpShape& shape, const hkTransform& transformToWorld )
{
	switch( shape.getType() ) 
	{

	// The basic convex shapes.
	
	case hkcdShapeType::SPHERE:
		{
			const hkpSphereShape& sphereShape = static_cast<const hkpSphereShape&>( shape );
			accumulateForcesOnSphere( sphereShape, transformToWorld );
			break;
		}


	case hkcdShapeType::CYLINDER:
		{
			const hkpCylinderShape& cylinderShape = static_cast<const hkpCylinderShape&>( shape );
			accumulateForcesOnCylinder( cylinderShape, transformToWorld );
			break;
		}

	case hkcdShapeType::TRIANGLE:
		{
			const hkpTriangleShape& triangleShape = static_cast<const hkpTriangleShape&>( shape );
			accumulateForcesOnTriangle( triangleShape, transformToWorld );
			break;
		}

	case hkcdShapeType::BOX:
		{
			const hkpBoxShape& boxShape = static_cast<const hkpBoxShape&>( shape );
			accumulateForcesOnBox( boxShape, transformToWorld );
			break;
		}

	case hkcdShapeType::CAPSULE:
		{
			const hkpCapsuleShape& capsuleShape = static_cast<const hkpCapsuleShape&>( shape );
			accumulateForcesOnCapsule( capsuleShape, transformToWorld );
			break;
		}

	case hkcdShapeType::CONVEX_VERTICES:
		{
			if ( m_obbFactor > 0.0f )
			{
				accumulateForcesOnObb( shape, transformToWorld );
			}
			else
			{
				const hkpConvexVerticesShape& cvShape = static_cast<const hkpConvexVerticesShape&>( shape );
				accumulateForcesOnConvexVertices( cvShape, transformToWorld );
			}
			break;
		}

	// Shapes with a single child shape.

	case hkcdShapeType::TRANSFORM:
		{
			const hkpTransformShape& transformShape = static_cast<const hkpTransformShape&>( shape );
			const hkpShape& childShape = *transformShape.getChildShape();
			hkTransform composedTransform;
			{
				composedTransform.setMul( transformToWorld, transformShape.getTransform() );
			}
			accumulateForcesOnShape( childShape, composedTransform );
			break;
		}

	case hkcdShapeType::CONVEX_TRANSFORM:
		{
			const hkpConvexTransformShape& transformShape = static_cast<const hkpConvexTransformShape&>( shape );
			const hkpShape& childShape = *transformShape.getChildShape();
			hkTransform localTransform; transformShape.getTransform( &localTransform );
			hkTransform composedTransform; composedTransform.setMul( transformToWorld, localTransform );
			accumulateForcesOnShape( childShape, composedTransform );
			break;
		}

	case hkcdShapeType::CONVEX_TRANSLATE:
		{
			const hkpConvexTranslateShape& translateShape = static_cast<const hkpConvexTranslateShape&>( shape );
			const hkpShape& childShape = *translateShape.getChildShape();
			hkVector4 newTranslation;
			{
				newTranslation.setRotatedDir( transformToWorld.getRotation(), translateShape.getTranslation() );
				newTranslation.add( transformToWorld.getTranslation() );
			}
			hkTransform composedTransform; composedTransform.set( transformToWorld.getRotation(), newTranslation );
			accumulateForcesOnShape( childShape, composedTransform );
			break;
		}

	case hkcdShapeType::BV:
		{
			const hkpBvShape& bvShape = static_cast<const hkpBvShape&>( shape );
			const hkpShape& childShape = *bvShape.getChildShape();
			accumulateForcesOnShape( childShape, transformToWorld );
			break;
		}

	// Composite shapes which implement the hkpContainer interface via hkpShapeCollection.

	case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION:
	case hkcdShapeType::COLLECTION:
	case hkcdShapeType::LIST:
		{
			const hkpShapeCollection& collection = static_cast<const hkpShapeCollection&>( shape );
			hkpShapeBuffer buffer;
			for ( hkpShapeKey k = collection.getFirstKey(); k != HK_INVALID_SHAPE_KEY; k = collection.getNextKey( k ) )
			{
				accumulateForcesOnShape( *collection.getChildShape( k, buffer ), transformToWorld );
			}
			break;
		}

	case hkcdShapeType::EXTENDED_MESH:
	case hkcdShapeType::TRIANGLE_COLLECTION:
		{
			const hkpShapeCollection& collection = static_cast<const hkpShapeCollection&>( shape );
			hkpShapeBuffer buffer;
			for ( hkpShapeKey k = collection.getFirstKey(); k != HK_INVALID_SHAPE_KEY; k = collection.getNextKey( k ) )
			{
				accumulateForcesOnShape( *collection.getChildShape( k, buffer ), transformToWorld );
			}
			break;
		}

	// Composite shapes which directly implement hkpContainer.

	case hkcdShapeType::CONVEX_LIST:
		{
			const hkpConvexListShape& convexListShape = static_cast<const hkpConvexListShape&>( shape );
			hkpShapeBuffer buffer;
			for ( hkpShapeKey k = convexListShape.getFirstKey(); k != HK_INVALID_SHAPE_KEY; k = convexListShape.getNextKey( k ) )
			{
				accumulateForcesOnShape( *convexListShape.getChildShape( k, buffer ), transformToWorld );
			}
			break;
		}

	// Composite shapes which can return an implementation of hkpContainer.
	
	case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE:
	case hkcdShapeType::BV_TREE:
	case hkcdShapeType::MOPP:
		{
			const hkpShapeContainer& container = *shape.getContainer();
			hkpShapeBuffer buffer;
			for ( hkpShapeKey k = container.getFirstKey(); k != HK_INVALID_SHAPE_KEY; k = container.getNextKey( k ) )
			{
				accumulateForcesOnShape( *container.getChildShape( k, buffer ), transformToWorld );
			}
			break;
		}

	
	case hkcdShapeType::CONVEX:

	// Ignore with respect to wind.
	case hkcdShapeType::PLANE:
	case hkcdShapeType::PHANTOM_CALLBACK:
	case hkcdShapeType::SAMPLED_HEIGHT_FIELD:
	case hkcdShapeType::MULTI_RAY:
	case hkcdShapeType::HEIGHT_FIELD:
	case hkcdShapeType::SPHERE_REP:

	// Deprecated shapes.
	case hkcdShapeType::CONVEX_PIECE:

	// Default is do nothing.
	default:
		{
			HK_WARN_ONCE(0x2fcb825e, "Wind applied to shape type which has no wind implementation.");
			break;
		}
	}
}

// ////////////////////////////////////////////////////////////////////////
// SPHERE
// ////////////////////////////////////////////////////////////////////////

void hkpWind::WindOnShape::accumulateForcesOnSphere( const hkpSphereShape& sphere, const hkTransform& transformToWorld )
{
	HK_TIME_CODE_BLOCK( "Sphere", HK_NULL );

	// We can ignore any rotations.
	hkVector4 centre = transformToWorld.getTranslation();

	hkVector4 windAtCentre;
	{
		getWindVector( centre, windAtCentre );
	}

	// Because we normalize, we can't handle zero length vectors.
	if ( windAtCentre.lengthSquared<3>().isGreaterZero() )
	{
		const hkSimdReal radius = hkSimdReal::fromFloat(sphere.getRadius());

		// Find point at which to apply the force.
		hkVector4 targetPoint;
		{
			hkVector4 radiusOpposingWind = windAtCentre;
			{
				radiusOpposingWind.normalize<3>();
				// the centroid of a hemispherical shell is half-way along the radius.
				radiusOpposingWind.setMul( -hkSimdReal_Inv2 * radius, radiusOpposingWind );
			}
			targetPoint.setAdd( centre, radiusOpposingWind );
		}

		hkVector4 drag;
		{
			drag.setMul( hkSimdReal::fromFloat((hkReal(2) / hkReal(3)) * HK_REAL_PI) * radius * radius, windAtCentre );
		}

		accumulate( targetPoint, drag );
	}
}

// ////////////////////////////////////////////////////////////////////////
// BOX
// ////////////////////////////////////////////////////////////////////////

void hkpWind::WindOnShape::accumulateForcesOnBox( const hkpBoxShape& box, const hkTransform& transformToWorld )
{	
	HK_TIME_CODE_BLOCK( "Box", HK_NULL );

	hkVector4 windAtCenter;
	{
		getWindVector( transformToWorld.getTranslation(), windAtCenter );
	}

	const hkVector4& halfExtents = box.getHalfExtents();

	hkVector4 hYZX; hYZX.setPermutation<hkVectorPermutation::YZXW>(halfExtents);
	hkVector4 hZXY; hZXY.setPermutation<hkVectorPermutation::ZXYW>(halfExtents);

	hkVector4 area4;
	area4.setMul(hYZX,hZXY); area4.mul(hkSimdReal_4);

	for ( int i = 0; i < 3; ++i )
	{
		// a vertex in the center of the face.
		hkVector4 faceVertex;
		{
			faceVertex.setMul(hkVector4::getConstant((hkVectorConstant)(HK_QUADREAL_1000+i)), halfExtents);
		}

		// Apply the rotation to the face to get normals for three of the faces.
		hkVector4 norm;
		{
			norm._setRotatedDir( transformToWorld.getRotation(), faceVertex );
			norm.normalize<3>();
		}

		const hkSimdReal windOnFace = norm.dot<3>( windAtCenter );

		hkVector4 drag;
		{
			drag.setMul(area4.getComponent(i) * windOnFace, norm );
		}

		hkVector4 targetPoint;
		{
			targetPoint.setFlipSign(faceVertex, windOnFace.greaterZero());
			targetPoint._setTransformedPos( transformToWorld, targetPoint );
		}

		accumulate( targetPoint, drag );
	}
}

// ////////////////////////////////////////////////////////////////////////
// TRIANGLE
// ////////////////////////////////////////////////////////////////////////

void hkpWind::WindOnShape::accumulateForcesOnTriangle( const hkpTriangleShape& triangle, const hkTransform& transformToWorld )
{
	HK_TIME_CODE_BLOCK( "Triangle", HK_NULL );

	const hkVector4& vertexA = triangle.getVertex<0>();
	const hkVector4& vertexB = triangle.getVertex<1>();
	const hkVector4& vertexC = triangle.getVertex<2>();

	hkVector4 edgeAB;
	{
		edgeAB.setSub( vertexB, vertexA );
	}
	hkVector4 edgeAC;
	{
		edgeAC.setSub( vertexC, vertexA );
	}

	hkVector4 normal;
	hkSimdReal area;
	{
		normal.setCross( edgeAB, edgeAC );
		area = hkSimdReal_Inv2 * normal.normalizeWithLength<3>();
		normal._setRotatedDir( transformToWorld.getRotation(), normal );
	}	

	// The centroid in world coords.
	hkVector4 centroid;
	{
		centroid.setAdd( edgeAB, edgeAC );
		centroid.mul(hkSimdReal_Inv3);
		centroid.add( vertexA );
		centroid._setTransformedPos( transformToWorld, centroid );
	}

	hkVector4 windAtCentroid;
	{
		getWindVector( centroid, windAtCentroid );
	}

	hkVector4 drag;
	{
		drag.setMul( area * windAtCentroid.dot<3>( normal ) , normal );
	}

	accumulate( centroid, drag );
}

// ////////////////////////////////////////////////////////////////////////
// WIND ON CYLINDER SIDE
// ////////////////////////////////////////////////////////////////////////

void hkpWind::WindOnShape::accumulateForcesOnCylinderSide( const hkVector4& worldA, const hkVector4& worldB, hkReal radius, hkVector4& normalAtAOut, hkVector4& windVectorOut )
{
	// In the degenerate case that the cylinder has no height and is therefore a sphere,
	// we can pick any normal we want.
	if ( worldA.allEqual<3>(worldB,hkSimdReal::fromFloat(1e-3f)) ) {
		getWindVector( worldA, windVectorOut );
		normalAtAOut = hkVector4::getConstant<HK_QUADREAL_0100>();
		return;
	}

	hkVector4 center;
	hkSimdReal height;
	{
		normalAtAOut.setSub( worldA, worldB );
		center.setAddMul( worldB, normalAtAOut,hkSimdReal_Inv2 );
		height = normalAtAOut.normalizeWithLength<3>();
	}

	getWindVector( center, windVectorOut );

	hkVector4 windAgainstSide;
	{
		windAgainstSide.setMul( windVectorOut.dot<3>( normalAtAOut ), normalAtAOut );
		windAgainstSide.setSub( windVectorOut, windAgainstSide );
	}

	const hkSimdReal radiusSr = hkSimdReal::fromFloat(radius);

	hkVector4 drag;
	{
		drag.setMul( hkSimdReal_PiOver2 * height * radiusSr, windAgainstSide );
	}

	// Find point at which to apply the force.
	hkVector4 targetPoint;
	{
		hkVector4 radiusOpposingWind = windAgainstSide;
		// An inaccurate target point is irrelevant if the wind is weak.
		radiusOpposingWind.normalizeIfNotZero<3>();
		// The centroid of the half-cylindrical surface is 2/pi along the radius.
		radiusOpposingWind.mul( -(hkSimdReal_2 / hkSimdReal_Pi) * radiusSr );
		targetPoint.setAdd( center, radiusOpposingWind );
	}

	accumulate( targetPoint, drag );
}


// ////////////////////////////////////////////////////////////////////////
// CYLINDER
// ////////////////////////////////////////////////////////////////////////

void hkpWind::WindOnShape::accumulateForcesOnCylinder( const hkpCylinderShape& cylinder, const hkTransform& transformToWorld )
{
	HK_TIME_CODE_BLOCK( "Cylinder", HK_NULL );

	hkVector4 worldA;
	{
		worldA._setTransformedPos( transformToWorld, cylinder.getVertex<0>() );
	}
	hkVector4 worldB;
	{
		worldB._setTransformedPos( transformToWorld, cylinder.getVertex<1>() );
	}

	const hkSimdReal radius = hkSimdReal::fromFloat(cylinder.getCylinderRadius());

	hkVector4 windAtCenter;
	hkVector4 normalAtA;
	accumulateForcesOnCylinderSide( worldA, worldB, cylinder.getCylinderRadius(), normalAtA, windAtCenter );

	const hkSimdReal area = hkSimdReal_Pi * radius * radius;

	const hkSimdReal windOnFaceA = windAtCenter.dot<3>( normalAtA );

	hkVector4 drag;
	{
		drag.setMul( area * windOnFaceA, normalAtA );
	}

	if ( windOnFaceA.isLessZero() )
	{
		accumulate( worldA, drag );
	}
	else
	{
		accumulate( worldB, drag );
	}

}

// ////////////////////////////////////////////////////////////////////////
// CAPSULE
// ////////////////////////////////////////////////////////////////////////

void hkpWind::WindOnShape::accumulateForcesOnCapsule( const hkpCapsuleShape& capsule, const hkTransform& transformToWorld )
{
	HK_TIME_CODE_BLOCK( "Capsule", HK_NULL );

	hkVector4 worldA;
	{
		worldA._setTransformedPos( transformToWorld, capsule.getVertex<0>() );
	}
	hkVector4 worldB;
	{
		worldB._setTransformedPos( transformToWorld, capsule.getVertex<1>() );
	}

	const hkSimdReal radius = hkSimdReal::fromFloat(capsule.getRadius());

	// The normal at end A.
	hkVector4 normal;
	hkVector4 windAtCenter;
	accumulateForcesOnCylinderSide( worldA, worldB, capsule.getRadius(), normal, windAtCenter );

	const hkSimdReal strengthOfWind = windAtCenter.length<3>();

	if ( strengthOfWind.isGreaterZero() )
	{
		// "Wind space" is a defined by the wind and normal at A.
		// i,j,k form an orthonormal basis for wind space *unless* the wind and
		// normal are parallel.
		hkVector4 i,j,k;
		{
			j.setNeg<3>( windAtCenter );
			j.normalize<3,HK_ACC_23_BIT,HK_SQRT_IGNORE>();
			k.setCross( j, normal );
			// We want to normalize k, but if we can't then it doesn't make a significant
			// difference to our calculations.
			k.normalizeIfNotZero<3>();
			i.setCross( j, k );
		}

		// This transforms (direction) vectors from wind to world space.
		hkMatrix3 toWorldSpace;
		{
			toWorldSpace.setCols( i, j, k );
		}

		// Psi is the angle between the wind and the normal at a.
		const hkSimdReal cosPsi = -normal.dot<3>( j );
		const hkSimdReal sinPsi = -normal.dot<3>( i );
		hkSimdReal psi;
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED) && !defined(HK_ARCH_ARM)
#if defined(HK_REAL_IS_DOUBLE)
		psi.m_real = hkMath::twoAcos(cosPsi.m_real);
#else
		psi.m_real = hkMath::quadAcos(cosPsi.m_real);
#endif
#else
		psi.setFromFloat( hkMath::acos( cosPsi.getReal() ) );
#endif

		{ // End A	
			hkVector4 drag;
			{
				
				hkVector4 forceInWindSpace; forceInWindSpace.set( sinPsi * sinPsi, -psi + sinPsi * cosPsi, hkSimdReal_0, hkSimdReal_0 );
				forceInWindSpace.mul( hkSimdReal::fromFloat(2.0f / 3.0f) * strengthOfWind * radius * radius );
				drag._setRotatedDir( toWorldSpace, forceInWindSpace );
			}

			hkVector4 targetPoint;
			{
				hkVector4 targetInWindSpace; targetInWindSpace.set( sinPsi, cosPsi - hkSimdReal_1, hkSimdReal_0, hkSimdReal_0);
				targetInWindSpace.mul(-hkSimdReal_Inv4 * radius );
				targetPoint._setRotatedDir( toWorldSpace, targetInWindSpace );
				targetPoint.add( worldA );
			}
			accumulate( targetPoint, drag );
		}

		{ // End B	
			// Same as A case except that:
			// psiAtB = pi - psiAtA (so sinPsiAtB = sinPsiAtA, cosPsiAtB = -cosPsiAtA)
			// iAtB = -iAtA
			// kAtB = -kAtB

			hkVector4 drag;
			{
				hkVector4 forceInWindSpace; forceInWindSpace.set( -(sinPsi * sinPsi), -(hkSimdReal_Pi - psi) - sinPsi * cosPsi, hkSimdReal_0, hkSimdReal_0);
				forceInWindSpace.mul( hkSimdReal::fromFloat(2.0f / 3.0f) * strengthOfWind * radius * radius );
				drag._setRotatedDir( toWorldSpace, forceInWindSpace );
			}

			hkVector4 targetPoint;
			{
				hkVector4 targetInWindSpace; targetInWindSpace.set( -sinPsi, -cosPsi - hkSimdReal_1, hkSimdReal_0, hkSimdReal_0);
				targetInWindSpace.mul( -hkSimdReal_Inv4 * radius );
				targetPoint._setRotatedDir( toWorldSpace, targetInWindSpace );
				targetPoint.add( worldB );
			}
			accumulate( targetPoint, drag );
		}		
	}
}

// ////////////////////////////////////////////////////////////////////////
// CONVEX VERTICES SHAPE
// ////////////////////////////////////////////////////////////////////////

void hkpWind::WindOnShape::accumulateForcesOnConvexVertices( const hkpConvexVerticesShape& cvShape, const hkTransform& transformToWorld )
{
	HK_TIME_CODE_BLOCK( "Convex verts", HK_NULL );

	hkVector4 windAtCenter;
	{
		hkVector4 center;
		{
			cvShape.getCentre( center );
			center._setTransformedPos( transformToWorld, center );
		}
		getWindVector( center, windAtCenter );
	}

	// Ensure there is a connectivity structure we can use.
	hkpConvexVerticesConnectivityUtil::ensureConnectivity( &cvShape );

	const hkpConvexVerticesConnectivity* connectivity = cvShape.getConnectivity();

	// Convert vertices all at once.
	// Assumption: getNumCollisionSpheres returns the number of vertices.
	const int numVertices = cvShape.getNumCollisionSpheres();
	hkLocalArray<hkVector4> vertArray( numVertices );
	{
		// Assumption: getOriginalVertices preserves the index order.
		cvShape.getOriginalVertices( vertArray );
	}

	// The first index of the current face.
	hkpVertexId faceIndex = 0;
	// The index of the face we're currently considering.
	hkpVertexId index;

	// Iterate over the faces:
	const int numFaces = connectivity->getNumFaces();
	for ( int face = 0; face < numFaces; ++face ) 
	{
		const hkpVertexId numVerticesThisFace = connectivity->m_numVerticesPerFace[ face ];

		// skip faces which aren't faces!
		if (  numVerticesThisFace < 3 )
		{
			faceIndex = faceIndex + numVerticesThisFace;
			continue;
		}

		const hkVector4 *const vertexA = &vertArray[ connectivity->m_vertexIndices[faceIndex] ];
		const hkVector4* vertexB = &vertArray[ connectivity->m_vertexIndices[faceIndex + 1] ];

		// This is not normalized during the calculation.
		hkVector4 normal;

		// Twice the total area of the face.
		hkSimdReal twiceArea; twiceArea.setZero();
		// The "weighted" centroid of the current face. To get the centroid, divide by (4 * twiceArea).
		hkVector4 weightedCentroid; weightedCentroid.setZero();

		index = faceIndex + 2;

		// Calculate the area and centroid of the current face.

		// Divide the face into quadrilaterals (requires about half the cross-products of a triangulation).
		
		
		

		while( index + 1 < faceIndex + numVerticesThisFace )
		{
			const hkVector4 *const vertexC = &vertArray[ connectivity->m_vertexIndices[index] ];
			const hkVector4 *const vertexD = &vertArray[ connectivity->m_vertexIndices[index + 1] ];

			hkVector4 ac;
			{
				ac.setSub( *vertexC, *vertexA );
			}
			hkVector4 bd;
			{
				bd.setSub( *vertexD, *vertexB );
			}

			hkSimdReal quadTwiceArea;
			{
				normal.setCross( ac, bd );
				quadTwiceArea = normal.normalizeWithLength<3,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>();
			}

			// We approximate the area-centroid by the nearby vertex-centroid.
			
			hkVector4 quadQuadrupleCentroid;
			{
				quadQuadrupleCentroid.setAdd( *vertexA, *vertexB );
				quadQuadrupleCentroid.add( *vertexC );
				quadQuadrupleCentroid.add( *vertexD );
			}

			twiceArea = twiceArea + quadTwiceArea;
			weightedCentroid.addMul( quadTwiceArea, quadQuadrupleCentroid );

			vertexB = vertexD;
			index += 2;
		}

		// The face may have a final triangle.
		if ( index + 1 == faceIndex + numVerticesThisFace )
		{
			const hkVector4 *const vertexC = &vertArray[ connectivity->m_vertexIndices[ index ] ];

			hkVector4 ab;
			{
				ab.setSub( *vertexB, *vertexA );
			}
			hkVector4 ac;
			{
				ac.setSub( *vertexC, *vertexA );
			}

			hkSimdReal triangleTwiceArea;
			{
				normal.setCross( ab, ac );
				triangleTwiceArea = normal.normalizeWithLength<3,HK_ACC_12_BIT,HK_SQRT_SET_ZERO>();
			}

			hkVector4 triangleQuadrupleCentroid;
			{
				// Adding two edges gives 3*centroid. We have to rescale it to match the 4*centroid
				// that the quadrilaterals produce.
				triangleQuadrupleCentroid.setAdd( ab, ac );
				triangleQuadrupleCentroid.mul( hkSimdReal::fromFloat(4.0f / 3.0f) );
				triangleQuadrupleCentroid.addMul( hkSimdReal_4, *vertexA);
			}

			twiceArea = twiceArea + triangleTwiceArea;
			weightedCentroid.addMul( triangleTwiceArea, triangleQuadrupleCentroid );

			++index;
		}

		// Apply forces to the face just considered.
		normal.setRotatedDir( transformToWorld.getRotation(), normal );

		hkVector4 centroid;
		{
			hkSimdReal invArea; invArea.setReciprocal(hkSimdReal_4 * twiceArea);
			centroid.setMul( invArea , weightedCentroid );
			centroid._setTransformedPos( transformToWorld, centroid );
		}

		hkSimdReal windOnFace;
		{
			windOnFace = windAtCenter.dot<3>( normal );
		}

		// Wind only pushes.
		if ( windOnFace.isLessZero() )
		{
			hkVector4 drag;	
			{
				drag.setMul( hkSimdReal_Inv2 * twiceArea * windOnFace, normal );
			}
			accumulate( centroid, drag );
		}

		faceIndex = index;
	}
}


// ////////////////////////////////////////////////////////////////////////
// AABB SHAPE
// ////////////////////////////////////////////////////////////////////////

//
// Draws a wireframe box using debugging lines.
//
void displayBox(const hkTransform& objectToWorld, const hkVector4& halfExtents)
{
	// points
	hkVector4 point[8];
	{
		for (int i = 0; i < 8; ++i)
		{
			hkVector4 v = halfExtents;
			if ( i & 1 )	v( 0 ) *= -1.0f;
			if ( i & 2 )	v( 1 ) *= -1.0f;
			if ( i & 4 )	v( 2 ) *= -1.0f;
			point[i].setTransformedPos( objectToWorld, v );
		}
	}

	// draw edges between points which differ in a single bit.
	{
		for ( int i = 0; i < 8; ++i )
		{
			for ( int bit = 1; bit < 8; bit <<= 1 )
			{
				int j = i ^ bit;
				if ( i < j )
				{
					HK_DISPLAY_LINE( point[i], point[j], hkColor::RED );
				}
			}
		}
	}
}

void hkpWind::WindOnShape::accumulateForcesOnObb( const hkpShape& shape, const hkTransform& transformToWorld )
{
	// This implementation uses an AABBs calculated in object space.
	hkAabb aabb;
	{
		hkTransform identity;
		identity.setIdentity();
		shape.getAabb( identity, 0.0f, aabb );
	}

	// Find the half-extents of the AABB.
	hkVector4 halfExtents;
	{
		halfExtents.setSub( aabb.m_max, aabb.m_min );
		halfExtents.mul( hkSimdReal_Inv2 );
	}

	hkTransform transformAabbToWorld;
	{
		hkTransform aabbToObject;
		{
			aabbToObject.setIdentity();
			hkVector4 centre;
			{
				centre.setSub( aabb.m_max, halfExtents );
			}
			aabbToObject.setTranslation( centre );
		}
		transformAabbToWorld.setMul( transformToWorld, aabbToObject );
	}

	halfExtents.mul( hkSimdReal::fromFloat(m_obbFactor) );

	//displayBox( transformAabbToWorld, halfExtents );

	// Call the box implementation to obtain the effect of the wind.
	hkpBoxShape box( halfExtents, 0.0f );
	accumulateForcesOnBox( box, transformAabbToWorld );
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
