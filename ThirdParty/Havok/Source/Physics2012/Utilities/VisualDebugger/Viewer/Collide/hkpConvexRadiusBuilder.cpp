/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpConvexRadiusBuilder.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>

#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Deprecated/MultiSphere/hkpMultiSphereShape.h>
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTransform/hkpConvexTransformShape.h>
#include <Physics2012/Collide/Shape/Misc/Bv/hkpBvShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/HeightField/SampledHeightField/hkpSampledHeightFieldShape.h>
#include <Physics2012/Collide/Shape/HeightField/Plane/hkpPlaneShape.h>

#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Misc/MultiRay/hkpMultiRayShape.h>

#include <Physics2012/Collide/Shape/Deprecated/ConvexPieceMesh/hkpConvexPieceShape.h>
#include <Physics2012/Collide/Util/hkpTriangleUtil.h>

#include <Common/Internal/ConvexHull/hkGeometryUtility.h>

#include <Common/Visualize/Shape/hkDisplayPlane.h>
#include <Common/Visualize/Shape/hkDisplaySphere.h>
#include <Common/Visualize/Shape/hkDisplayCylinder.h>
#include <Common/Visualize/Shape/hkDisplayCapsule.h>
#include <Common/Visualize/Shape/hkDisplayBox.h>
#include <Common/Visualize/Shape/hkDisplayConvex.h>
#include <Common/Visualize/hkDebugDisplay.h>

#include <Physics2012/Utilities/VisualDebugger/Viewer/hkpShapeContinueData.h>

hkpConvexRadiusBuilder::hkpConvexRadiusBuilderEnvironment::hkpConvexRadiusBuilderEnvironment()
:	m_minVisibleRadius(0.001f)
{
}



hkpConvexRadiusBuilder::hkpConvexRadiusBuilder(const hkpConvexRadiusBuilderEnvironment& env)
:	m_environment(env),
	m_currentGeometry(HK_NULL)
{
}


void hkpConvexRadiusBuilder::buildDisplayGeometries(		const hkpShape* shape,
														hkArray<hkDisplayGeometry*>& displayGeometries)
{

	hkTransform transform;
	transform.setIdentity();

	resetCurrentRawGeometry();
	displayGeometries.clear();

 	buildShapeDisplay(shape, transform, displayGeometries);
}

static hkBool _createConvexDisplayFromPlanes( const hkArray<hkVector4>& planeEqs, const hkTransform& transform, hkGeometry* outputGeometry )
{
	// Create verts from planes
	hkArray<hkVector4> verts;
	hkGeometryUtility::createVerticesFromPlaneEquations(planeEqs, verts);

	if (verts.getSize() < 1)
	{
		return false;
	}

	// Transform verts
	hkArray<hkVector4> transformedVerts;
	int numVerts = verts.getSize();
	transformedVerts.setSize(numVerts);
	for (int i=0; i < numVerts; ++i)
	{
		transformedVerts[i].setTransformedPos(transform, verts[i]);
	}

	hkStridedVertices stridedVerts;
	{
		stridedVerts.m_numVertices = transformedVerts.getSize();
		stridedVerts.m_striding = sizeof(hkVector4);
		stridedVerts.m_vertices = &(transformedVerts[0](0));
	}

	hkGeometryUtility::createConvexGeometry( stridedVerts, *outputGeometry );

	return true;
}

HK_DISABLE_OPTIMIZATION_VS2008_X64
static hkSimdReal _getPlaneToSurfaceDistance(const hkpConvexVerticesShape* shape)
{
	const hkArray<hkVector4>& planes = shape->getPlaneEquations();
	hkArray<hkVector4> vertices;
	shape->getOriginalVertices( vertices );

	const int numVerts = vertices.getSize();
	const int numPlanes = planes.getSize();
		
	hkSimdReal closest = hkSimdReal::fromFloat(-1000.0f);
	for (int v = 0; v < numVerts; v++)
	{
		const hkVector4& vert = vertices[v];
		for (int p = 0; p < numPlanes; p++)
		{
			const hkSimdReal distFromPlane = vert.dot<3>(planes[p]) + planes[p].getW();
			if (distFromPlane > closest) // dot is negative if on internal side.
			{
				closest = distFromPlane;
			}
		}
	}
	return closest;
}
HK_RESTORE_OPTIMIZATION_VS2008_X64

static void _addBoundingPlanes(const hkpShape* s, hkReal extraRadius, hkArray<hkVector4>& planes)
{
	hkAabb aabb;
	s->getAabb(hkTransform::getIdentity(), extraRadius, aabb);
	planes.expandBy(6);
	
	int numPlanes = planes.getSize();
	planes[ numPlanes - 6 ].set( 1, 0, 0, -aabb.m_max(0) );
	planes[ numPlanes - 5 ].set( 0, 1, 0, -aabb.m_max(1) );
	planes[ numPlanes - 4 ].set( 0, 0, 1, -aabb.m_max(2) );
	planes[ numPlanes - 3 ].set( -1, 0, 0, aabb.m_min(0) );
	planes[ numPlanes - 2 ].set( 0, -1, 0, aabb.m_min(1) );
	planes[ numPlanes - 1 ].set( 0, 0, -1, aabb.m_min(2) );
}

void hkpConvexRadiusBuilder::buildShapeDisplay_ShapeContainer( const hkpShapeContainer* shapeContainer, const hkTransform& transform, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	hkpShapeBuffer buffer;
	for (hkpShapeKey key = shapeContainer->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = shapeContainer->getNextKey( key ) )
	{
		const hkpShape* child = shapeContainer->getChildShape(key, buffer );
		buildShapeDisplay(child, transform, displayGeometries);
	}
}

void hkpConvexRadiusBuilder::buildShapeDisplay_ConvexPiece( const hkpConvexPieceShape* triangulatedConvexShape, const hkTransform& transform, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	if (triangulatedConvexShape->getRadius() > m_environment.m_minVisibleRadius)
	{
		hkpShapeBuffer buffer2;
		for ( int i = 0 ; i < triangulatedConvexShape->m_numDisplayShapeKeys ; i++ )
		{
			const hkpTriangleShape& triangleShape = *( static_cast< const hkpTriangleShape* >( 
				triangulatedConvexShape->m_displayMesh->getChildShape( triangulatedConvexShape->m_displayShapeKeys[i], buffer2 ) ));
			buildShapeDisplay(&triangleShape, transform, displayGeometries);	
		}
	}
}

void hkpConvexRadiusBuilder::buildShapeDisplay_Cylinder( const hkpCylinderShape* s, const hkTransform& transform, hkArray<hkDisplayGeometry*>& displayGeometries, const hkVector4* scale )
{
	const hkSimdReal cylRadius = hkSimdReal::fromFloat(s->getCylinderRadius());
	const hkSimdReal convexRadius = hkSimdReal::fromFloat(s->getRadius()); // cyl code has this as the 'padding' radius
	if (convexRadius > hkSimdReal::fromFloat(m_environment.m_minVisibleRadius))
	{
		hkVector4 top = s->getVertex<1>();
		hkVector4 bottom = s->getVertex<0>();

		// add in the top and bottom radius
		// the radius on the sides will be added to the cyl radius
		hkVector4 axis; axis.setSub(top, bottom); axis.normalize<3>();
		axis.mul(convexRadius);
		top.add(axis);
		bottom.sub(axis);

		hkSimdReal totalRadius = cylRadius + convexRadius;

		// Apply scale if present
		if (scale)
		{
			const hkSimdReal scale0 = scale->getComponent<0>();
			HK_ON_DEBUG(hkVector4 scaleX; scaleX.setAll(scale0);)
			HK_WARN_ON_DEBUG_IF(!scale->allExactlyEqual<3>(scaleX), 0x39d9a3a3, "Shape type not supported with non-uniform scale");
			hkSimdReal absScale0; absScale0.setAbs(scale0);

			top.mul(scale0);
			bottom.mul(scale0);
			totalRadius = totalRadius*absScale0;
		}

		hkDisplayCylinder* displayCylinder = new hkDisplayCylinder( top, bottom, totalRadius.getReal() );
		displayCylinder->setTransform( transform );
		displayGeometries.pushBack( displayCylinder );
	}
}

void hkpConvexRadiusBuilder::buildShapeDisplay_Box( const hkpBoxShape* boxShape, const hkTransform& transform, hkArray<hkDisplayGeometry*>& displayGeometries, const hkVector4* scale )
{
	const hkSimdReal convexRadius = hkSimdReal::fromFloat(boxShape->getRadius());
	if (convexRadius > hkSimdReal::fromFloat(m_environment.m_minVisibleRadius))
	{
		hkVector4 trueExtents; trueExtents.setAll(convexRadius);
		trueExtents.add( boxShape->getHalfExtents() );
		if (scale)
		{
			trueExtents.mul(*scale);
		}
		hkDisplayBox* displayBox = new hkDisplayBox(trueExtents);
		displayBox->setTransform( transform );
		displayGeometries.pushBack(displayBox);
	}
}

void hkpConvexRadiusBuilder::buildShapeDisplay_Triangle( const hkpTriangleShape* triangleShape, const hkTransform& transform, hkArray<hkDisplayGeometry*>& displayGeometries, const hkVector4* scale )
{
	hkSimdReal convexRadius = hkSimdReal::fromFloat(triangleShape->getRadius());
	if (convexRadius > hkSimdReal::fromFloat(m_environment.m_minVisibleRadius))
	{
		// A triangle with radius is a convex object made of 5 planes (3 edges, 2 faces)
		// but the radius is distance from the verts so really the pointy edges are rounded
		// so we will add 3 planes at each end point too. If we don't thin triangled get very pointy
		// ends that look like a bug to the unknowing user. 

		// For a large landscape this may be too much to show (as each tri is now up to around 12 new tris (10 normal + flatened pointy ends)
		// but most landscapes have 0 radius.
		hkVector4 p0 = triangleShape->getVertex<0>();
		hkVector4 p1 = triangleShape->getVertex<1>();
		hkVector4 p2 = triangleShape->getVertex<2>();

		if (scale)
		{
			const hkSimdReal scale0 = scale->getComponent<0>();
			HK_ON_DEBUG(hkVector4 scaleX; scaleX.setAll(scale0);)
			HK_WARN_ON_DEBUG_IF(!scale->allExactlyEqual<3>(scaleX), 0x39d9a3a3, "Shape type not supported with non-uniform scale");
			hkSimdReal absScale0; absScale0.setAbs(scale0);
			
			p0.mul(scale0);
			p1.mul(scale0);
			p2.mul(scale0);
			convexRadius = convexRadius*absScale0;
		}


		if ( hkpTriangleUtil::isDegenerate(p0, p1, p2, 0.001f) )
			return;

		hkVector4 edge01; edge01.setSub(p1, p0);
		edge01.normalize<3>();
		hkVector4 edge12; edge12.setSub(p2, p1);
		edge12.normalize<3>();
		hkVector4 edge02; edge02.setSub(p2, p0);
		edge02.normalize<3>();
		hkVector4 normal; normal.setCross(edge01, edge02); 

		hkArray<hkVector4> planes(8);
		planes.setSize(8);				
		hkVector4 planePoint; 

		// face planes
		planes[0] = normal;
		planePoint.setAddMul(p0, planes[0], convexRadius);
		planes[0].setXYZ_W(normal, -planes[0].dot<3>( planePoint ));

		planes[1].setNeg<4>(normal); 
		planePoint.setAddMul(p0, planes[1], convexRadius);
		planes[1].setW(-planes[1].dot<3>( planePoint ));

		// edge planes
		planes[2].setCross(edge01, normal); 
		planePoint.setAddMul(p0, planes[2], convexRadius);
		planes[2].setW(-planes[2].dot<3>( planePoint ));

		planes[3].setCross(edge12, normal); 
		planePoint.setAddMul(p1, planes[3], convexRadius);
		planes[3].setW(-planes[3].dot<3>( planePoint ));

		planes[4].setCross(normal, edge02); 
		planePoint.setAddMul(p2, planes[4], convexRadius);
		planes[4].setW(-planes[4].dot<3>( planePoint ));

		// extra edges and end points to tighten it up.
		planes[5].setAdd(planes[2], planes[3]); planes[5].normalize<3>();
		planePoint.setAddMul(p1, planes[5], convexRadius);
		planes[5].setW(-planes[5].dot<3>( planePoint ));

		planes[6].setAdd(planes[3], planes[4]); planes[6].normalize<3>();
		planePoint.setAddMul(p2, planes[6], convexRadius);
		planes[6].setW(-planes[6].dot<3>( planePoint ));

		planes[7].setAdd(planes[4], planes[2]); planes[7].normalize<3>();
		planePoint.setAddMul(p0, planes[7], convexRadius);
		planes[7].setW(-planes[7].dot<3>( planePoint ));

		// create vertices based on planes intersection points
		hkDisplayGeometry* displayGeom = getCurrentRawGeometry( displayGeometries );
		if ( !_createConvexDisplayFromPlanes( planes, transform, displayGeom->getGeometry() ) )
		{		
			HK_WARN(0x3236452A, "Could not create shape representing the convex radius around a triangle!");
		}
	}
}

void hkpConvexRadiusBuilder::buildShapeDisplay_ConvexVertices( const hkpConvexVerticesShape* convexVerticesShape, const hkTransform& transform, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	const hkSimdReal convexRadius = hkSimdReal::fromFloat(convexVerticesShape->getRadius());
	if (convexRadius > hkSimdReal::fromFloat(m_environment.m_minVisibleRadius))
	{
		// copy plane equations and expand
		const hkArray<hkVector4>& planeEqs = convexVerticesShape->getPlaneEquations();

		if (planeEqs.getSize() > 0)
		{
			// We don't know if the planes are from the verts or are pre expanded by the 
			// convex radius (as happens with the shrink filter)
			// Once there is agreement on whether the planes should
			// always be a radius distance from the verts then 
			hkSimdReal currentDist = _getPlaneToSurfaceDistance(convexVerticesShape); // may be negative

			const hkSimdReal RADIUS_EPSILON = hkSimdReal::fromFloat(0.005f);
			hkArray<hkVector4> newPlanesEqs; newPlanesEqs = planeEqs;
			if ( currentDist > (RADIUS_EPSILON - convexRadius) ) // then not already pushed out (vert within radius of a plane), assume the planes are close to 0 distance from the real object surface 
			{
				int numPlanes = planeEqs.getSize();
				for (int i=0; i < numPlanes; ++i)
				{
					const hkSimdReal cr = newPlanesEqs[i].getW() - convexRadius;
					newPlanesEqs[i].setW(cr);
				}
			}

			_addBoundingPlanes(convexVerticesShape, convexRadius.getReal(), newPlanesEqs);

			hkDisplayGeometry* displayGeom = getCurrentRawGeometry( displayGeometries );
			if ( !_createConvexDisplayFromPlanes( newPlanesEqs, transform, displayGeom->getGeometry() ) )
			{		
				HK_WARN(0x3236452A, "Could not create shape representing the convex radius around a convex shape!");
			}
		}
		else
		{
			// If we don't have any plane equations we should build the geometry from the collision spheres instead
			// to stay as close to the actual shape as possible.

			const int numSpheres = convexVerticesShape->getNumCollisionSpheres();

			hkLocalArray<hkSphere> vertices(numSpheres); vertices.setSize( numSpheres );

			const hkSphere* spheres = convexVerticesShape->getCollisionSpheres(vertices.begin());
			
			// Convert these vertices to the transformed space.
			hkArray<hkVector4> transformedVertices;
			transformedVertices.setSize( numSpheres );
			for(int i = 0; i < numSpheres; i++)
			{
				transformedVertices[i].setTransformedPos(transform, spheres[i].getPosition());
			}

			hkGeometry* outputGeom = new hkGeometry;

			// HVK-5032
			hkGeometryUtility::createConvexGeometry(transformedVertices,*outputGeom);

			hkDisplayConvex* displayGeom = new hkDisplayConvex(outputGeom);

			displayGeometries.pushBack(displayGeom);
		}
	}
}

hkBool hkpConvexRadiusBuilder::buildShapeDisplay_UserShapes( const hkpShape* shape, const hkTransform& transform, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	hkBool handled = false;
	for (int i = 0; i < hkpUserConvexRadiusBuilder::getInstance().m_userConvexRadiusBuilders.getSize(); ++i )
	{
		if ( hkpUserConvexRadiusBuilder::getInstance().m_userConvexRadiusBuilders[i].type == shape->getType() )
		{
			hkpUserConvexRadiusBuilder::getInstance().m_userConvexRadiusBuilders[i].f( shape, transform, displayGeometries, this );
			handled = true;
			continue;
		}
	}
	return handled;
}

// This is the alternative to having a buildDisplayGeometry as a virtual function in Shape.
void hkpConvexRadiusBuilder::buildShapeDisplay_ConvexTransform( const hkpConvexTransformShape* shape, const hkTransform& parentTransform, hkArray<hkDisplayGeometry*>& displayGeometries)
{
	const hkQsTransform& convexTransform = shape->getQsTransform();			
	const hkVector4& childScale = convexTransform.getScale();
	hkTransform convexTransformNoScale; convexTransform.copyToTransformNoScale(convexTransformNoScale);
	hkTransform childTransform; childTransform.setMul(parentTransform, convexTransformNoScale);

	const hkpShape* childShape = shape->getChildShape();

	if (childScale.allEqual<3>( hkVector4::getConstant<HK_QUADREAL_1>(), hkSimdReal_Eps ))//if uniform scale
	{
		buildShapeDisplay(shape->getChildShape(), childTransform, displayGeometries, HK_NULL);
	}

	else
	{
		switch (childShape->getType())
		{
		case hkcdShapeType::CONVEX_VERTICES:
			{
				// create a new convex vertices shape with its vertices scaled.
				const hkpConvexVerticesShape* convexVerticesShape = static_cast<const hkpConvexVerticesShape*>(childShape);

				// copy the untransformed vertices 
				hkArray<hkVector4> vertices; convexVerticesShape->getOriginalVertices(vertices);

				// apply the transform to these vertices
				for ( hkArray<hkVector4>::iterator i = vertices.begin(); i!=vertices.end(); i++ )
				{
					shape->transformVertex( (*i), &(*i) );
					//(*i)._setTransformedPos(transform,*i);
				}

				// make sure no shrinking takes place (we want the shape exactly as it is)
				hkpConvexVerticesShape::BuildConfig config;
				{
					config.m_convexRadius = convexVerticesShape->getRadius();
					config.m_shrinkByConvexRadius = false;
				}

				const hkStridedVertices strided(vertices);
				const hkpConvexVerticesShape newShape(strided,config);
				buildShapeDisplay_ConvexVertices( &newShape, hkTransform::getIdentity(), displayGeometries );

				break;
			}
		default:
			{
				buildShapeDisplay(childShape, childTransform, displayGeometries, &childScale);
				break;
			}
		}
	}
}

// This is the alternative to having a buildDisplayGeometry as a virtual function in Shape.
void hkpConvexRadiusBuilder::buildShapeDisplay(		const hkpShape* shape,
													const hkTransform& transform,
													hkArray<hkDisplayGeometry*>& displayGeometries,
													const hkVector4* scale)
{
	switch (shape->getType())
	{
		//
		// These do not use convex radius:
		//
		case hkcdShapeType::SPHERE:
		case hkcdShapeType::MULTI_SPHERE:
		case hkcdShapeType::PLANE:
		case hkcdShapeType::CAPSULE:
		case hkcdShapeType::MULTI_RAY:
		case hkcdShapeType::SAMPLED_HEIGHT_FIELD:
			break;
		
		//
		// Shape wrapper types
		//

		case hkcdShapeType::CONVEX_TRANSLATE:
		{
			const hkpConvexTranslateShape* ts = static_cast<const hkpConvexTranslateShape*>( shape );
			

			hkVector4 translation = ts->getTranslation();
			if (scale)
			{
				translation.mul(*scale);				
			}
			hkTransform tst; tst.setIdentity();
			tst.setTranslation(translation);
			hkTransform T; T.setMul( transform, tst );
			buildShapeDisplay( ts->getChildShape(), T, displayGeometries, scale);
			break;
		}
		case hkcdShapeType::CONVEX_TRANSFORM:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpConvexTransformShape* ts = static_cast<const hkpConvexTransformShape*>( shape );
			buildShapeDisplay_ConvexTransform(ts, transform, displayGeometries);
			break;

		}	
		case hkcdShapeType::TRANSFORM:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpTransformShape* ts = static_cast<const hkpTransformShape*>( shape );
			hkTransform T; T.setMul( transform, ts->getTransform() );
			buildShapeDisplay( ts->getChildShape(), T, displayGeometries);
			break;
		}
		case hkcdShapeType::BV:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpBvShape* bvShape = static_cast<const hkpBvShape*>(shape);
			buildShapeDisplay( bvShape->getBoundingVolumeShape(), transform, displayGeometries);
			break;
		}
		case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE:
		case hkcdShapeType::BV_TREE:
		case hkcdShapeType::MOPP:
		{
			const hkpBvTreeShape* bvShape = static_cast<const hkpBvTreeShape*>(shape);
			const hkpShapeContainer* shapeContainer = bvShape->getContainer();
			buildShapeDisplay_ShapeContainer( shapeContainer, transform, displayGeometries );
			break;
		}
		case hkcdShapeType::CONVEX_LIST:
		case hkcdShapeType::LIST:
		case hkcdShapeType::COLLECTION:
		case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION:
		case hkcdShapeType::TRIANGLE_COLLECTION:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpShapeContainer* container = shape->getContainer();
			buildShapeDisplay_ShapeContainer( container, transform, displayGeometries );
			break;
		}
		case hkcdShapeType::CONVEX_PIECE:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpConvexPieceShape* triangulatedConvexShape = static_cast<const hkpConvexPieceShape*>(shape);
			buildShapeDisplay_ConvexPiece( triangulatedConvexShape, transform, displayGeometries );
			break;
		}

		//
		// shapes that use a radius
		//

		case hkcdShapeType::CYLINDER:
		{
			const hkpCylinderShape* s = static_cast<const hkpCylinderShape*>(shape);
			buildShapeDisplay_Cylinder( s, transform, displayGeometries, scale );
			break;
		}
		case hkcdShapeType::BOX:
		{
			const hkpBoxShape* boxShape = static_cast<const hkpBoxShape*>(shape);
			buildShapeDisplay_Box( boxShape, transform, displayGeometries, scale );
			break;
		}
		case hkcdShapeType::TRIANGLE:
		{
			const hkpTriangleShape* triangleShape = static_cast<const hkpTriangleShape*>(shape);
			buildShapeDisplay_Triangle( triangleShape, transform, displayGeometries, scale );
			break;
		}
		case hkcdShapeType::CONVEX_VERTICES:
		{
			const hkpConvexVerticesShape* convexVerticesShape = static_cast<const hkpConvexVerticesShape*>(shape);
			buildShapeDisplay_ConvexVertices( convexVerticesShape, transform, displayGeometries ); // scale can only be from convextransformshape which will already be accounted for by this point?
			break;
		}
		default:
			buildShapeDisplay_UserShapes( shape, transform, displayGeometries );
			break;
	}
}

hkDisplayGeometry* hkpConvexRadiusBuilder::getCurrentRawGeometry(hkArray<hkDisplayGeometry*>& displayGeometries)
{
	if (m_currentGeometry == HK_NULL)
	{
		hkGeometry* geom = new hkGeometry;
		m_currentGeometry = new hkDisplayConvex(geom);
		displayGeometries.pushBack(m_currentGeometry);
	}
	return m_currentGeometry;
}


void hkpConvexRadiusBuilder::resetCurrentRawGeometry()
{
	m_currentGeometry = HK_NULL;
}


HK_SINGLETON_IMPLEMENTATION(hkpUserConvexRadiusBuilder);


void hkpUserConvexRadiusBuilder::registerUserConvexRadiusDisplayBuilder( ConvexRadiusBuilderFunction f, hkpShapeType type )
{
	for (int i = 0; i < m_userConvexRadiusBuilders.getSize(); ++i )
	{
		if ( m_userConvexRadiusBuilders[i].type == type )
		{
			HK_WARN(0x7bbfa3c4, "You have registered two convex shape display builders for user type" << type << ". Do you have two different shapes with this type?");
			return;
		}
	}
	UserShapeBuilder b;
	b.f = f;
	b.type = type;

	m_userConvexRadiusBuilders.pushBack(b);
}


void hkpConvexRadiusBuilder::buildDisplayGeometries( const hkReferencedObject* source, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	buildDisplayGeometries( static_cast<const hkpShape*>( source ), displayGeometries );
}


hkReferencedObject* hkpConvexRadiusBuilder::getInitialContinueData( const hkReferencedObject* source )
{
	return new hkpShapeContinueData();
}

hkBool hkpConvexRadiusBuilder::buildPartialDisplayGeometries( const hkReferencedObject* source, int& numSimpleShapes, hkReferencedObject* continueData, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	hkTransform transform;
	transform.setIdentity();

	resetCurrentRawGeometry();
	displayGeometries.clear();

	const hkpShape *const shape = static_cast<const hkpShape*>( source );
	hkpShapeContinueData *const shapeContinueData = static_cast<hkpShapeContinueData*>( continueData );

	if ( buildPartialShapeDisplay( shape, transform, 0, numSimpleShapes, shapeContinueData, displayGeometries ) )
	{
		// We've finished with this continue data.
		shapeContinueData->removeReference();
		return true;
	}
	else
	{
		return false;
	}
}

hkBool hkpConvexRadiusBuilder::buildPartialShapeDisplay_ShapeContainer( const hkpShapeContainer* shapeContainer, const hkTransform& transform, int branchDepth, int& numSimpleShapes, hkpShapeContinueData* continueData, hkArray<hkDisplayGeometry*>& displayGeometries, const hkVector4* scale )
{
	HK_ASSERT2( 0x3e421e73, numSimpleShapes > 0, "Cannot build a shape display for 0 numSimpleShapes." );
	const int totalDepthSoFar = continueData->m_shapeKeys.getSize();
	HK_ASSERT2( 0x3e421e82, branchDepth <= totalDepthSoFar, "The continue data is inconsistent with the traversal of the shape hierarchy." );
	if ( totalDepthSoFar == branchDepth )
	{
		// This is the first time we've processed this shape:
		continueData->m_shapeKeys.expandOne() = shapeContainer->getFirstKey();
	}
	hkpShapeKey key = continueData->m_shapeKeys[branchDepth];
	hkpShapeBuffer buffer;

	while ( ( key != HK_INVALID_SHAPE_KEY ) && ( numSimpleShapes > 0 ) )
	{
		const hkpShape* child = shapeContainer->getChildShape(key, buffer );
		if ( buildPartialShapeDisplay( child, transform, branchDepth + 1, numSimpleShapes, continueData, displayGeometries, scale ) )
		{
			// We successfully processed the subshape, so try the next one.
			key = shapeContainer->getNextKey( key );
			continueData->m_shapeKeys[branchDepth] = key;
		}
		else
		{
			return false;
		}
	}

	if ( key == HK_INVALID_SHAPE_KEY )
	{
		// We finished processing this subshape.
		continueData->m_shapeKeys.popBack();
		return true;
	}
	else
	{
		return false;
	}
}

void hkpConvexRadiusBuilder::buildPartialShapeDisplay_ConvexTransform( const hkpConvexTransformShape* convexTransformShape, const hkTransform& parentTransform, int branchDepth, int& numSimpleShapes, hkpShapeContinueData* continueData, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	const hkQsTransform& convexTransform = convexTransformShape->getQsTransform();			
	const hkVector4& childScale = convexTransform.getScale();
	hkTransform convexTransformNoScale; convexTransform.copyToTransformNoScale(convexTransformNoScale);
	hkTransform childTransform; childTransform.setMul(parentTransform, convexTransformNoScale);

	const hkpShape* childShape = convexTransformShape->getChildShape();

	if (childScale.allEqual<3>( hkVector4::getConstant<HK_QUADREAL_1>(), hkSimdReal_Eps ))//if uniform scale
	{
		buildPartialShapeDisplay(childShape, childTransform, branchDepth + 1, numSimpleShapes, continueData, displayGeometries, HK_NULL);
	}

	else
	{
		switch (childShape->getType())
		{
		case hkcdShapeType::CONVEX_VERTICES:
			{
				// create a new convex vertices shape with its vertices scaled.
				const hkpConvexVerticesShape* convexVerticesShape = static_cast<const hkpConvexVerticesShape*>(childShape);

				// copy the untransformed vertices 
				hkArray<hkVector4> vertices; convexVerticesShape->getOriginalVertices(vertices);

				// apply the transform to these vertices
				for ( hkArray<hkVector4>::iterator i = vertices.begin(); i!=vertices.end(); i++ )
				{
					convexTransformShape->transformVertex( (*i), &(*i) );
					//(*i)._setTransformedPos(transform,*i);
				}

				// make sure no shrinking takes place (we want the shape exactly as it is)
				hkpConvexVerticesShape::BuildConfig config;
				{
					config.m_convexRadius = convexVerticesShape->getRadius();
					config.m_shrinkByConvexRadius = false;
				}

				const hkStridedVertices strided(vertices);
				const hkpConvexVerticesShape newShape(strided,config);
				buildShapeDisplay_ConvexVertices( &newShape, hkTransform::getIdentity(), displayGeometries );

				break;
			}
		default:
			{
				buildPartialShapeDisplay(childShape, childTransform, branchDepth + 1, numSimpleShapes, continueData, displayGeometries, &childScale);
				break;
			}
		}
	}
}

hkBool hkpConvexRadiusBuilder::buildPartialShapeDisplay( const hkpShape* shape, const hkTransform& transform, int branchDepth, int& numSimpleShapes, hkpShapeContinueData* continueData, hkArray<hkDisplayGeometry*>& displayGeometries, const hkVector4* scale )
{
	HK_ASSERT2( 0x3e421e73, numSimpleShapes > 0, "Cannot build a shape display for 0 numSimpleShapes." );

	switch( shape->getType() )
	{
		case hkcdShapeType::SPHERE:
		case hkcdShapeType::MULTI_SPHERE:
		case hkcdShapeType::PLANE:
		case hkcdShapeType::CAPSULE:
		case hkcdShapeType::MULTI_RAY:
		case hkcdShapeType::SAMPLED_HEIGHT_FIELD:
			// None of the above shapes use convex radius.
			break;
		case hkcdShapeType::CONVEX_TRANSLATE:
		{
			const hkpConvexTranslateShape* ts = static_cast<const hkpConvexTranslateShape*>( shape );
			
			hkVector4 translation = ts->getTranslation();
			if (scale)
			{
				translation.mul(*scale);				
			}
			hkTransform tst; tst.setIdentity();	tst.setTranslation( translation );
			hkTransform T; T.setMul( transform, tst );
			return buildPartialShapeDisplay( ts->getChildShape(), T, branchDepth, numSimpleShapes, continueData, displayGeometries );
		}

		case hkcdShapeType::CONVEX_TRANSFORM:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");
			const hkpConvexTransformShape* ts = static_cast<const hkpConvexTransformShape*>( shape );
			buildPartialShapeDisplay_ConvexTransform( ts, transform, branchDepth, numSimpleShapes, continueData, displayGeometries );
			break;
		}	

		case hkcdShapeType::TRANSFORM:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");
			const hkpTransformShape* ts = static_cast<const hkpTransformShape*>( shape );
			hkTransform T; T.setMul( transform, ts->getTransform() );
			return buildPartialShapeDisplay( ts->getChildShape(), T, branchDepth, numSimpleShapes, continueData, displayGeometries );
		}
		case hkcdShapeType::BV:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");
			const hkpBvShape* bvShape = static_cast<const hkpBvShape*>(shape);
			return buildPartialShapeDisplay( bvShape->getBoundingVolumeShape(), transform, branchDepth, numSimpleShapes, continueData, displayGeometries );
		}
		case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE:
		case hkcdShapeType::BV_TREE:
		case hkcdShapeType::MOPP:
		{
			const hkpBvTreeShape* bvShape = static_cast<const hkpBvTreeShape*>(shape);
			const hkpShapeContainer* shapeContainer = bvShape->getContainer();
			return buildPartialShapeDisplay_ShapeContainer( shapeContainer, transform, branchDepth, numSimpleShapes, continueData, displayGeometries );
		}
		case hkcdShapeType::CONVEX_LIST:
		case hkcdShapeType::LIST:
		case hkcdShapeType::COLLECTION:
		case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION:
		case hkcdShapeType::TRIANGLE_COLLECTION:
		{
			const hkpShapeContainer* container = shape->getContainer();
			return buildPartialShapeDisplay_ShapeContainer( container, transform, branchDepth, numSimpleShapes, continueData, displayGeometries, scale );
		}
		case hkcdShapeType::CONVEX_PIECE:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");
			const hkpConvexPieceShape* triangulatedConvexShape = static_cast<const hkpConvexPieceShape*>(shape);
			// This shape is deprecated, so we haven't implemented partial building for it.
			buildShapeDisplay_ConvexPiece( triangulatedConvexShape, transform, displayGeometries );
			--numSimpleShapes;
			break;
		}
		case hkcdShapeType::CYLINDER:
		{
			const hkpCylinderShape* s = static_cast<const hkpCylinderShape*>(shape);
			buildShapeDisplay_Cylinder( s, transform, displayGeometries, scale );
			--numSimpleShapes;
			break;
		}
		case hkcdShapeType::BOX:
		{
			const hkpBoxShape* boxShape = static_cast<const hkpBoxShape*>(shape);
			buildShapeDisplay_Box( boxShape, transform, displayGeometries, scale );
			--numSimpleShapes;
			break;
		}
		case hkcdShapeType::TRIANGLE:
		{
			const hkpTriangleShape* triangleShape = static_cast<const hkpTriangleShape*>(shape);
			buildShapeDisplay_Triangle( triangleShape, transform, displayGeometries, scale );
			--numSimpleShapes;
			break;
		}
		case hkcdShapeType::CONVEX_VERTICES:
		{
			const hkpConvexVerticesShape* convexVerticesShape = static_cast<const hkpConvexVerticesShape*>(shape);
			buildShapeDisplay_ConvexVertices( convexVerticesShape, transform, displayGeometries ); // scale can only be from convextransformshape which will already be accounted for by this point?
			--numSimpleShapes;
			break;
		}
		default:
		{
			if ( buildShapeDisplay_UserShapes( shape, transform, displayGeometries ) )
			{
				--numSimpleShapes;
			}
			break;
		}
	}
	return true;
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
