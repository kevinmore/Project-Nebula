/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Util/ShapeShrinker/hkpShapeShrinker.h>
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTransform/hkpConvexTransformShape.h>
#include <Physics2012/Collide/Shape/Misc/Bv/hkpBvShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/hkpBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Deprecated/ConvexList/hkpConvexListShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesConnectivityUtil.h>

#include <Common/Internal/ConvexHull/hkGeometryUtility.h>
#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>

// Rebuild plane equations from connectivity considering convex radius
static inline
void _buildNewPlaneEquations( hkpConvexVerticesConnectivity* connectivity, const hkArray< hkVector4 >& vertices , hkReal radius, hkArray< hkVector4 >& planeEquations )
{
	planeEquations.reserveExactly( connectivity->getNumFaces() );

	int faceStart = 0;
	for ( int i = 0; i < connectivity->getNumFaces(); ++i )
	{
		int numFaceIndices = connectivity->m_numVerticesPerFace[ i ];
		if ( numFaceIndices < 3 )
		{
			connectivity->m_numVerticesPerFace.removeAtAndCopy(i);
			i--;
			faceStart += numFaceIndices;
			continue;
		}

		// Pick the first non-degenerated indices tuple
		hkBool32	res = hkFalse32;
		hkVector4	n; n.setZero();
		hkVector4	origin; origin.setZero();

		for(int i0=0; res==hkFalse32 && i0<numFaceIndices; ++i0)
		{
			for(int i1=i0+1; res==hkFalse32 && i1<numFaceIndices; ++i1)
			{
				for(int i2=i1+1; res==hkFalse32 && i2<numFaceIndices; ++i2)
				{
					int idx0 = connectivity->m_vertexIndices[ faceStart + i0 ];
					int idx1 = connectivity->m_vertexIndices[ faceStart + i1 ];
					int idx2 = connectivity->m_vertexIndices[ faceStart + i2 ];

					hkVector4 v0 = vertices[ idx0 ];
					hkVector4 v1 = vertices[ idx1 ];
					hkVector4 v2 = vertices[ idx2 ];

					hkVector4 e1, e2;
					e1.setSub( v1, v0 );
					e2.setSub( v2, v0 );

					n.setCross( e1, e2 );
					origin	=	v0;
					res		=	n.normalizeIfNotZero<3>();
				}
			}
		}
		
		faceStart += numFaceIndices;

		if (res == hkFalse32)
		{
			HK_WARN(0xabba9999,"Found colinear vertices when building plane equation. Using this plane might cause random collision failure.");
			
			n.setZero();
			origin.setZero();
		}

		n.setW( -( n.dot<3>( origin ) + hkSimdReal::fromFloat(radius) ) );

		planeEquations.pushBackUnchecked( n );
	}
}


// Compute mid half extent if AABB
static HK_FORCE_INLINE
hkSimdReal _getMidHalfExtent( const hkAabb& aabb )
{
	hkVector4 extent;
	extent.setSub( aabb.m_max, aabb.m_min );

	int minIndex = extent.getIndexOfMinComponent<3>();
	int maxIndex = extent.getIndexOfMaxComponent<3>();

	if ( ( (minIndex == 0) && (maxIndex == 2) ) ||
		 ( (minIndex == 2) && (maxIndex == 0) ) )
	{
		return hkSimdReal_Inv2 * extent.getComponent<1>();
	}
	else if ( ( (minIndex == 1) && (maxIndex == 2) ) ||
			  ( (minIndex == 2) && (maxIndex == 1) ) )
	{
		return hkSimdReal_Inv2 * extent.getComponent<0>();
	}
	else
	{
		return hkSimdReal_Inv2 * extent.getComponent<2>();
	}
}


// Compute smallest distance from a point to a plane
static HK_FORCE_INLINE
hkSimdReal _getMinDistance( const hkArray< hkVector4 >& planes, hkVector4Parameter point )
{
	hkSimdReal minDistance = hkSimdReal_Max;
	for ( int i = 0; i < planes.getSize(); ++i )
	{
		const hkVector4 plane = planes[ i ];

		const hkSimdReal distance = plane.dot4xyz1( point );

		minDistance.setMin( minDistance, -distance );
	}

	return minDistance;
}

static hkpConvexVerticesShape* _shrinkConvexVerticesShape( const hkpConvexVerticesShape* convexVerticesShape, hkReal radius, hkBool optimize)
{	
	if(optimize)
	{
		// Built convex hull from shape vertices
		hkArray<hkVector4>					vertices; convexVerticesShape->getOriginalVertices(vertices);
		hkgpConvexHull						hull; hull.build(vertices);
		if(hull.getDimensions()==3)
		{
			// Scale by shifting planes by -radius
			hkgpConvexHull::AbsoluteScaleConfig	absConfig;
			absConfig.m_method	=	hkgpConvexHull::AbsoluteScaleConfig::SKM_PLANES;
			hull.absoluteScale(-radius, absConfig);

			if(hull.getDimensions()==3)
			{
				// Decimate vertices
				hkgpConvexHull*	smpHull = hull.clone();
				if(!smpHull->decimateVertices(vertices.getSize(),true))
				{
					delete smpHull;
					smpHull = hull.clone();
					HK_REPORT("Failed to decimate");
				}

				// Retrieve new vertices.
				vertices.clear();
				smpHull->fetchPositions(hkgpConvexHull::INTERNAL_VERTICES, vertices);
				delete smpHull;

				// Built a new hkpConvexVerticeShape against these new vertices.
				hkpConvexVerticesShape::BuildConfig	buildConfig;
				buildConfig.m_convexRadius			=	convexVerticesShape->getRadius();
				buildConfig.m_shrinkByConvexRadius	=	false;
				buildConfig.m_createConnectivity	=	true;

				return new hkpConvexVerticesShape(vertices,buildConfig);
			}
		} else HK_WARN_ALWAYS(0x679D17FA, "Cannot use optimized shrinking on non-volumetric(3D) shapes, falling back to legacy method.");
	}
	
	// Legacy shrinking method, used if optimize is false or the shape is not 3D.
	{
		// Cut shape by moved planes
		const hkArray< hkVector4 >& planeEq = convexVerticesShape->getPlaneEquations();

		const hkpConvexVerticesShape* newShape = convexVerticesShape;
		newShape->addReference();			

		for ( int pidx = 0; pidx < planeEq.getSize(); ++pidx )
		{
			// Move plane
			hkVector4 cutPlane = planeEq[ pidx ];

			// Cut
			const hkpConvexVerticesShape* cutShape = newShape;
			newShape = hkpConvexVerticesConnectivityUtil::cut( cutShape, cutPlane, radius, HK_REAL_EPSILON );

			// Destroy old shape
			cutShape->removeReference();

			// Validate new shape
			if ( newShape == HK_NULL )
			{
				return HK_NULL;
			}
		}

		return const_cast< hkpConvexVerticesShape* >(  newShape  );
	}
}

static hkReal _findConvexVerticesShapeMaxDisplacement( const hkpConvexVerticesShape* newShape, const hkpConvexVerticesShape* oldShape, hkVector4Parameter centroid )
{
	// Get original vertices
	hkArray< hkVector4 > newVertices;
	newShape->getOriginalVertices( newVertices );
	if ( newVertices.getSize() < 4)
	{
		return hkReal(0);
	}

	
	// Use support() function to find maximum displacement of vertices
	hkSimdReal maxSquaredDisplacement; maxSquaredDisplacement.setZero();

	for ( int i = 0; i < newVertices.getSize(); ++i )
	{
		hkVector4 direction;
		direction.setSub( newVertices[ i ], centroid );
		direction.normalize<3>();

		hkcdVertex newSupportingVertex;
		newShape->getSupportingVertex( direction, newSupportingVertex );

		hkcdVertex oldSupportingVertex;
		oldShape->getSupportingVertex( direction, oldSupportingVertex );

		hkVector4 offset;
		offset.setSub( oldSupportingVertex, newSupportingVertex );

		const hkSimdReal squaredDistance = offset.lengthSquared<3>();
		
		maxSquaredDisplacement.setMax( squaredDistance, maxSquaredDisplacement );	
	}

	return maxSquaredDisplacement.sqrt().getReal();
}

static hkpShape* _shrinkShape(hkpShape* shape, hkArray<hkpShapeShrinker::ShapePair>& doneShapes, hkBool optimize)
{
	// To support shared shapes in a hierarchy, we check if we have done this one before.
	for (int dsi=0; dsi < doneShapes.getSize(); ++dsi)
	{
		if (doneShapes[dsi].originalShape == shape)
		{
			return doneShapes[dsi].newShape;
		}
	}

	hkpShapeShrinker::ShapePair ds;
	ds.originalShape = shape;
	ds.newShape = HK_NULL;

	switch (shape->getType())
	{
		// types that require no shrink (radius == proper radius or no convex shape radius used)
	case hkcdShapeType::SPHERE: 
	case hkcdShapeType::MULTI_SPHERE:
	case hkcdShapeType::PLANE:
	case hkcdShapeType::CAPSULE: 
			break;

		// Unshrinkable (2D)
	case hkcdShapeType::TRIANGLE:
	case hkcdShapeType::TRIANGLE_COLLECTION:
			break;

		// Collections or shape wrappers
	case hkcdShapeType::CONVEX_TRANSLATE:
		{
			hkpConvexTranslateShape* ts = static_cast<hkpConvexTranslateShape*>( shape );
			
			// Grab the child shape and shrink it
			hkpConvexShape* shrunkenChild = static_cast<hkpConvexShape*>(_shrinkShape(const_cast<hkpConvexShape*>(ts->getChildShape()), doneShapes, optimize));

			// EXP-685 : It can be NULL if there were no changes
			if (shrunkenChild)
			{
				// Create a new translate shape with the newly shrunken child
				hkpConvexTranslateShape* shrunkenTranslateShape = new hkpConvexTranslateShape(shrunkenChild, ts->getTranslation());

				ds.newShape = shrunkenTranslateShape;
			}
			break;
		}
	case hkcdShapeType::CONVEX_TRANSFORM:
		{
			hkpConvexTransformShape* ts = static_cast<hkpConvexTransformShape*>( shape );

			// Grab the child shape and shrink it
			hkpConvexShape* shrunkenChild = static_cast<hkpConvexShape*>(_shrinkShape(const_cast<hkpConvexShape*>(ts->getChildShape()), doneShapes, optimize));

			// EXP-685 : It can be NULL if there were no changes
			if (shrunkenChild)
			{
				// Create a new transform shape with the newly shrunken child
				hkTransform localTransform; ts->getTransform( &localTransform );
				hkpConvexTransformShape* shrunkenTransformShape = new hkpConvexTransformShape(shrunkenChild, localTransform);

				ds.newShape = shrunkenTransformShape;
			}

			break;
		}	
	case hkcdShapeType::TRANSFORM:
		{
			hkpTransformShape* ts = static_cast<hkpTransformShape*>( shape );

			// Grab the child shape and shrink it
			hkpShape* shrunkenChild = static_cast<hkpShape*>(_shrinkShape(const_cast<hkpShape*>(ts->getChildShape()), doneShapes, optimize));

			// EXP-685 : It can be NULL if there were no changes
			if (shrunkenChild)
			{
				// Create a new transform shape with the newly shrunken child
				hkpTransformShape* shrunkenTransformShape = new hkpTransformShape(shrunkenChild, ts->getTransform());

				ds.newShape = shrunkenTransformShape;
			}

			break;
		}
	case hkcdShapeType::BV:
		{
			hkpBvShape* bvShape = static_cast<hkpBvShape*>(shape);
			ds.newShape = _shrinkShape( const_cast<hkpShape*>(bvShape->getChildShape()), doneShapes , optimize);
			break;
		}

	case hkcdShapeType::BV_TREE:
	case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE:
	case hkcdShapeType::MOPP:
		{
			// const hkpBvTreeShape* bvShape = static_cast<const hkpBvTreeShape*>(shape);
			// TODO: could add an option to reduced the landscape radius. (can't 
			//       really add to radius as the MOPP is created? )

			break;
		}
	case hkcdShapeType::CONVEX_LIST:
	case hkcdShapeType::LIST:
	case hkcdShapeType::COLLECTION:
	case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION:
		{
			const hkpShapeContainer* shapeContainer = shape->getContainer();
			HK_ASSERT2(0xDBC03771, shapeContainer, "Shape to be shrunk returned a null shape container.");

			hkpShapeBuffer buffer;

			const bool isMutable = ((shape->getType() == hkcdShapeType::LIST) || (shape->getType() == hkcdShapeType::CONVEX_LIST));
			
            hkArray<hkpShape*> newShapes;
			newShapes.reserveExactly(shapeContainer->getNumChildShapes());
			
            bool foundNewOnes = false;
			for (hkpShapeKey key = shapeContainer->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = shapeContainer->getNextKey( key ) )
			{
				const hkpShape* child = shapeContainer->getChildShape(key, buffer );
				hkpShape* newShape = _shrinkShape(const_cast<hkpShape*>(child), doneShapes, optimize);
				if (newShape)
				{
					// ehh.. not in charge of the shapes in the collection so can't change them
					if (!isMutable)
					{
						HK_WARN(0x47b3d83b, "Found a shape collection with children that required changing. Not processing.");
					}
					foundNewOnes = true;
				}
				newShapes.pushBack(newShape? newShape : const_cast<hkpShape*>(child) );
			}

			hkpShape* newS = HK_NULL;
			if (foundNewOnes && isMutable)
			{
				if (shape->getType() == hkcdShapeType::LIST) 
				{
                    newS = new hkpListShape( newShapes.begin(), newShapes.getSize() );

                    // HVK-5588: if we created a new shape instances here, remove the reference that's added when the list shape is created
                    hkpListShape* newListShape = static_cast<hkpListShape*>(newS);
                    hkpListShape* oldListShape = static_cast<hkpListShape*>(shape);
                    for (int i = 0; i < newShapes.getSize(); ++i)
                    {
                        newListShape->m_childInfo[i].m_collisionFilterInfo = oldListShape->m_childInfo[i].m_collisionFilterInfo;
                        
                        if (newListShape->m_childInfo[i].m_shape != oldListShape->m_childInfo[i].m_shape) 
                        {   
                            newListShape->m_childInfo[i].m_shape->removeReference();
                        }
                    }
				}
				else if (shape->getType() == hkcdShapeType::CONVEX_LIST)
				{
					newS = new hkpConvexListShape( (const hkpConvexShape**)newShapes.begin(), newShapes.getSize() );
				}  

			}

			ds.newShape = newS;
			break;
		}

	case hkcdShapeType::CYLINDER: // created with two radii. One is the correct one, the other (padding) left at default, so if we reduce the normal 
		// radius by the padding, and the length by the padding too it will be visually correct
		{
			hkpCylinderShape* cylShape = static_cast<hkpCylinderShape*>(shape);
			hkReal rP = cylShape->getRadius();	
			hkSimdReal rPs = hkSimdReal::fromFloat(rP);
			hkReal rC = cylShape->getCylinderRadius();
			
			// the cylinder radius must exceed the padding, or shrinking will reduce the cylinder to zero volume
			bool validRadius = (rC > rP);

			// the cylinder length must exceed twice the padding, or shrinking will produce a zero length cylinder
			hkBool32 validLength;
			{
				hkVector4 d; d.setSub(cylShape->getVertex<0>(), cylShape->getVertex<1>());
				validLength = d.length<3>().isGreater(rPs + rPs);
			}
			
			if ( validRadius && validLength )
			{
				rC -= rP;
				
				cylShape->setCylinderRadius(rC);

				// shift the end points down by radius;
				hkVector4 dir; dir.setSub( cylShape->getVertex<1>(), cylShape->getVertex<0>() );
				dir.normalize<3>();
				dir.mul(rPs);

				hkVector4 newV0; newV0.setAdd(cylShape->getVertex<0>(), dir);
				hkVector4 newV1; newV1.setSub(cylShape->getVertex<1>(), dir );
				cylShape->setVertex<0>(newV0);
				cylShape->setVertex<1>(newV1);
			}
			else
			{
				HK_WARN_ALWAYS(0xabbafa2c, "Cylinder shape too small compared to 'extra radius' - unable to shrink.");
				break;
			}
			
			// EXP-682 : We need to register in-place changes as well, otherwise they may happen more than once
			ds.newShape = cylShape;
			break;
		}

	case hkcdShapeType::BOX: // exported from max etc with default radius so definitely needs a shrink
		{
			hkpBoxShape* boxShape = static_cast<hkpBoxShape*>(shape);

			// Reduce the extents by the radius
			hkVector4 ext = boxShape->getHalfExtents();
			
			// Get radius and adjust it
			const hkSimdReal minExtentHalf = ext.horizontalMin<3>() * hkSimdReal_Inv2;
			
			hkReal radius = boxShape->getRadius();
			if (radius > minExtentHalf.getReal())
			{
				HK_WARN_ALWAYS(0xabbafa2d, "Box shape too small compared to 'extra radius' - radius changed from " << radius << " to " << minExtentHalf.getReal() << ".");
				radius = minExtentHalf.getReal();
			}

			hkSimdReal deltaExt;
			deltaExt.setFromFloat( radius );

			ext.setSub( ext, deltaExt );

		
			boxShape->setHalfExtents( ext );
			boxShape->setRadius( radius );
			
			
			// EXP-682 : We need to register in-place changes as well, otherwise they may happen more than once
			ds.newShape = boxShape;
			break;
		}

	case hkcdShapeType::CONVEX_VERTICES:
		{
			hkpConvexVerticesShape* convexVerticesShape = static_cast< hkpConvexVerticesShape* >( shape );	

			if ( convexVerticesShape->getRadius() <= hkReal(0) )
			{
				convexVerticesShape->setRadius( hkReal(0) );
				break;
			}

			bool hadConnectivity = convexVerticesShape->getConnectivity() != HK_NULL;
			hkpConvexVerticesConnectivityUtil::ensureConnectivity( convexVerticesShape );

			const hkReal originalRadius = convexVerticesShape->getRadius();

			// Try to shrink
			hkpConvexVerticesShape* newShape = _shrinkConvexVerticesShape( convexVerticesShape, convexVerticesShape->getRadius(), optimize );
			if ( newShape == HK_NULL )
			{
				convexVerticesShape->setRadius( hkReal(0) );
				if (!hadConnectivity)
				{
					convexVerticesShape->setConnectivity(HK_NULL);
				}
				break;
			}

			// Compute center of mass of convex hull
			hkVector4 centerOfMass; centerOfMass.setZero();
			{
				hkArray<hkVector4> vertices;
				convexVerticesShape->getOriginalVertices(vertices);

				hkgpConvexHull hull;
				hull.build(vertices);
				if (hull.getDimensions() == 3)
				{
					hull.buildMassProperties();
					centerOfMass = hull.getCenterOfMass();
				}
			}

			// Validate shrinking
			hkReal maxDisplacement = _findConvexVerticesShapeMaxDisplacement( newShape, convexVerticesShape, centerOfMass );
			const hkReal kAllowedDisplacement = hkReal(2) * convexVerticesShape->getRadius();
			if ( maxDisplacement > kAllowedDisplacement )
			{
				// Compute new radius
				hkReal newRadius = hkReal(0.9f) * kAllowedDisplacement / maxDisplacement * convexVerticesShape->getRadius();
				convexVerticesShape->setRadiusUnchecked( newRadius );

				// Try to shrink again
				newShape->removeReference();
				newShape = _shrinkConvexVerticesShape( convexVerticesShape, convexVerticesShape->getRadius(), optimize );
				if ( newShape == HK_NULL )
				{
					HK_WARN_ALWAYS( 0xdd12ee34, "Suspicious convex vertices shape" );
					convexVerticesShape->setRadiusUnchecked( hkReal(0) );
					if (!hadConnectivity)
					{
						convexVerticesShape->setConnectivity(HK_NULL);
					}
					break;
				}
			}

			if (!hkMath::equal(newShape->getRadius(), originalRadius, hkReal(0.001f)))
			{
				HK_WARN_ALWAYS(0xabbafa2e, "Convex vertices shape too small compared to 'extra radius' - radius changed from " << originalRadius << " to " << newShape->getRadius() << ".");
			}

			// Overwrite plane equations
			const hkArray< hkVector4 >& planeEquations = convexVerticesShape->getPlaneEquations();

			// HKV-5061
			#if 1
			if ( planeEquations.getSize() >= newShape->getConnectivity()->getNumFaces() )
			{
				newShape->setPlaneEquations( planeEquations );
			}
			else
			{
				hkArray< hkVector4 > vertices;
				newShape->getOriginalVertices( vertices );

				hkArray< hkVector4 > newPlaneEquations;
				hkpConvexVerticesConnectivity* connectivity = const_cast<hkpConvexVerticesConnectivity*>(newShape->getConnectivity());
				_buildNewPlaneEquations( connectivity, vertices, newShape->getRadius(), newPlaneEquations );

				newShape->setPlaneEquations( newPlaneEquations );
			}
			#else
			newShape->setPlaneEquations( planeEquations );
			#endif

			if (!hadConnectivity)
			{
				convexVerticesShape->setConnectivity(HK_NULL);
				newShape->setConnectivity(HK_NULL);
			}

			ds.newShape = newShape;
			break;
		}

		//
		// Unhandled at this time
		//
	case hkcdShapeType::CONVEX_PIECE:
	case hkcdShapeType::SAMPLED_HEIGHT_FIELD:	
	default:
		break;
	}

	// HVK-3576
	// If we have created a new shape, copy the old user data to the new shape.
	if (ds.newShape)
	{
		ds.newShape->setUserData(ds.originalShape->getUserData());
		doneShapes.pushBack(ds);
	}

	return ds.newShape; // new shape, null if not new
}

hkpShape* hkpShapeShrinker::shrinkByConvexRadius( hkpShape* s, hkArray<ShapePair>* doneShapes , hkBool optimize)
{
	if (doneShapes)
	{
		return _shrinkShape(s, *doneShapes, optimize);
	}
	else
	{
		// make a temp one
		hkArray<ShapePair> shapeCache;
		return _shrinkShape(s, shapeCache, optimize);
	}
}


hkpBoxShape* hkpShapeShrinker::shrinkBoxShape( hkpBoxShape* boxShape, hkReal relShrinkRadius, hkReal allowedDisplacement )
{
	// Compute shrink radius
	hkVector4 halfExtents = boxShape->getHalfExtents();
	const hkSimdReal minDistance = halfExtents.horizontalMin<3>();

	hkAabb aabb;
	aabb.m_min.setNeg<3>( halfExtents );
	aabb.m_max = halfExtents;
	const hkSimdReal midHalfExtent = _getMidHalfExtent( aabb );

	hkSimdReal shrinkRadius; 
	shrinkRadius.setMin( hkSimdReal_Inv2 * minDistance, hkSimdReal::fromFloat(relShrinkRadius) * midHalfExtent );
	shrinkRadius.setMin( shrinkRadius, hkSimdReal::fromFloat(allowedDisplacement) ); 

	// Reduce the extents by the radius
	hkVector4 extent = boxShape->getHalfExtents();
	HK_ASSERT( 0xddf2f3a2, extent( 0 ) > shrinkRadius.getReal() && extent( 1 ) > shrinkRadius.getReal() && extent( 2 ) > shrinkRadius.getReal() );

	hkVector4 radius;
	radius.setZero(); radius.setXYZ( shrinkRadius );
	
	extent.sub( radius );
	boxShape->setHalfExtents( extent );


	// Overwrite convex radius
	hkReal newRadius = shrinkRadius.getReal() + boxShape->getRadius();
	boxShape->setRadius( newRadius );
	
	return boxShape;
}


hkpCylinderShape* hkpShapeShrinker::shrinkCylinderShape( hkpCylinderShape* cylinderShape, hkReal relShrinkRadius, hkReal allowedDisplacement )
{
	hkReal radius = cylinderShape->getCylinderRadius();
	const hkSimdReal radiusSr = hkSimdReal::fromFloat(radius);

	hkVector4 vertex0 = cylinderShape->getVertex<0>();
	hkVector4 vertex1 = cylinderShape->getVertex<1>();

	hkVector4 direction;
	direction.setSub( vertex1, vertex0 );
	const hkSimdReal halfHeight = direction.normalizeWithLength<3>() * hkSimdReal_Inv2;

	hkSimdReal minDistance; minDistance.setMin( radiusSr, halfHeight );
	hkSimdReal midHalfExtent; midHalfExtent.setMin( radiusSr, halfHeight );

	hkSimdReal shrinkRadius; shrinkRadius.setMin( hkSimdReal_Inv2 * minDistance, hkSimdReal::fromFloat(relShrinkRadius) * midHalfExtent );
	HK_ASSERT( 0xdd12eea2, radiusSr > shrinkRadius && halfHeight > shrinkRadius );
	
	
	// Reduce the radius 
	cylinderShape->setCylinderRadius( radius - shrinkRadius.getReal() );

	
	// Reduce the height
	vertex0.addMul( shrinkRadius, direction );
	vertex1.subMul( shrinkRadius, direction );

	cylinderShape->setVertex<0>( vertex0 );
	cylinderShape->setVertex<1>( vertex1 );


	// Overwrite convex radius
	hkReal newRadius = shrinkRadius.getReal() + cylinderShape->getRadius();
	cylinderShape->setRadius( newRadius );

	return cylinderShape;
}

hkpConvexVerticesShape* hkpShapeShrinker::shrinkConvexVerticesShape( hkpConvexVerticesShape* originalConvexVerticesShape, hkReal maximumConvexRadius, hkReal relShrinkRadius, hkReal allowedDisplacement, const char* shapeName , hkBool optimize)
{
	// Compute center of mass of convex hull
	hkVector4 centerOfMass; centerOfMass.setZero();
	{
		hkArray<hkVector4> vertices;
		originalConvexVerticesShape->getOriginalVertices(vertices);
		if(vertices.getSize()<4)
		{
			HK_WARN_ALWAYS(94545,"Shape shrinker does not support convex hull dimemsion less than 3");
			return(HK_NULL);
		}

		hkgpConvexHull hull;
		hull.build(vertices);
		hkSimdReal volume; volume.setZero();
		if (hull.getDimensions() == 3)
		{
			hull.buildMassProperties();
			centerOfMass = hull.getCenterOfMass();
			volume = hull.getVolume();
		}
		if (volume.isEqualZero())
		{
			return HK_NULL;
		}
	}

	const hkArray< hkVector4 >& planeEqns = originalConvexVerticesShape->getPlaneEquations(); 
	hkSimdReal minDistance = _getMinDistance( planeEqns, centerOfMass );
	if ( minDistance.isLessZero() )
	{
		if ( shapeName )
		{
			HK_WARN_ALWAYS( 0xabba3475, "Shape '" << shapeName << "' : Center Of Mass seems to be outside the object. Looks like the convex hull of the object is corrupted. ("<<minDistance.getReal()<<")");
		}
		else
		{
			HK_WARN_ALWAYS( 0xabba3465, "The Center Of Mass seems to be outside the object. Looks like the convex hull of the object is corrupted. ("<<minDistance.getReal()<<")");
		}
		minDistance.setZero();
	}

	hkAabb aabb;
	originalConvexVerticesShape->getAabb( hkTransform::getIdentity(), hkReal(0), aabb );
	const hkSimdReal midHalfExtent = _getMidHalfExtent( aabb );

	hkSimdReal shrinkRadius; 
	shrinkRadius.setMin( hkSimdReal_Inv2 * minDistance, hkSimdReal::fromFloat(relShrinkRadius) * midHalfExtent );
	shrinkRadius.setMin( shrinkRadius, hkSimdReal::fromFloat(maximumConvexRadius) );
	

	// Connectivity is needed for shrinking
	bool hadConnectivity = originalConvexVerticesShape->getConnectivity() != HK_NULL;
	hkpConvexVerticesConnectivityUtil::ensureConnectivity( originalConvexVerticesShape );

	// Try to shrink
	hkpConvexVerticesShape* newShape = _shrinkConvexVerticesShape( originalConvexVerticesShape, shrinkRadius.getReal() , optimize);
	if ( newShape == HK_NULL )
	{
		if (!hadConnectivity)
		{
			originalConvexVerticesShape->setConnectivity(HK_NULL);
		}
		return HK_NULL;
	}

	// Validate shrinking
	hkReal maxDisplacement = _findConvexVerticesShapeMaxDisplacement( newShape, originalConvexVerticesShape, centerOfMass );

	if ( maxDisplacement > allowedDisplacement )
	{
		// Compute new radius
		hkReal shrinkR = hkReal(0.9f) * allowedDisplacement / maxDisplacement * shrinkRadius.getReal();

		// Try to shrink again
		newShape->removeReference();
		newShape = _shrinkConvexVerticesShape( originalConvexVerticesShape, shrinkR, optimize);
		if ( newShape == HK_NULL )
		{
			if (!hadConnectivity)
			{
				originalConvexVerticesShape->setConnectivity(HK_NULL);
			}
			return HK_NULL;
		}

		shrinkRadius.setFromFloat(shrinkR);
	}

	// Overwrite convex radius
	hkReal newRadius = shrinkRadius.getReal() /*+ originalConvexVerticesShape->getRadius()*/;
	newShape->setRadiusUnchecked( newRadius );

	// Overwrite plane equations
	const hkArray< hkVector4 >& planeEquations = originalConvexVerticesShape->getPlaneEquations();

	if ( planeEquations.getSize() >= newShape->getConnectivity()->getNumFaces() )
	{
		newShape->setPlaneEquations( planeEquations );
	}
	else
	{
		hkArray< hkVector4 > vertices;
		newShape->getOriginalVertices( vertices );

		hkArray< hkVector4 > newPlaneEquations;
		hkpConvexVerticesConnectivity* connectivity = const_cast<hkpConvexVerticesConnectivity*>(newShape->getConnectivity());
		_buildNewPlaneEquations( connectivity, vertices, newShape->getRadius(), newPlaneEquations );

		newShape->setPlaneEquations( newPlaneEquations );
	}

	if (!hadConnectivity)
	{
		originalConvexVerticesShape->setConnectivity(HK_NULL);
		newShape->setConnectivity(HK_NULL);
	}

	return newShape;
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
