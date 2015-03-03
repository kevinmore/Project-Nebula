/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Utilities/Collide/ShapeUtils/ShapeScaling/hkpShapeScalingUtility.h>

#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTransform/hkpConvexTransformShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/ExtendedMeshShape/hkpExtendedMeshShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/SimpleMesh/hkpSimpleMeshShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Physics2012/Collide/Shape/Misc/Bv/hkpBvShape.h>

#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppUtility.h>


static hkpShape* _scaleShape( hkpShape* shape, hkSimdRealParameter sscale, hkArray<hkpShapeScalingUtility::ShapePair>& doneShapes )
{
	// To support shared shapes in a hierarchy, we check if we have done this one before.
	for (int dsi=0; dsi < doneShapes.getSize(); ++dsi)
	{
		if (doneShapes[dsi].originalShape == shape)
		{
			return doneShapes[dsi].newShape;
		}
	}
		
	hkpShapeScalingUtility::ShapePair doneShape;
	doneShape.originalShape = shape;
	doneShape.newShape = HK_NULL;
	
	// Switch on the shape type
	hkpShapeType type = shape->getType();
	switch ( type )
	{
	case hkcdShapeType::SPHERE:
		{
			hkpSphereShape* sphere = static_cast< hkpSphereShape* >( shape );

			// Scale the radius
			sphere->setRadius( sphere->getRadius() * sscale.getReal() );

			doneShape.newShape = sphere;
			break;
		}

	case hkcdShapeType::CYLINDER:
		{
			hkpCylinderShape* cylinder = static_cast< hkpCylinderShape* >( shape );

			// Scale the radius
			cylinder->setCylinderRadius( sscale.getReal() * cylinder->getCylinderRadius() );

			// Scale the endpoints
			for ( int i = 0; i < 2; i++ )
			{
				hkVector4 p;
				p = cylinder->getVertex( i );
				p.mul( sscale );
				cylinder->setVertex( i, p );
			}

			doneShape.newShape = cylinder;
			break;
		}

	case hkcdShapeType::TRIANGLE:
		{
			hkpTriangleShape* triangle = static_cast< hkpTriangleShape* >( shape );
			hkVector4 v;

			// Scale each vertex
			for ( int i = 0; i < 3; i++ )
			{
				v = triangle->getVertex( i );
				v.mul(sscale );
				triangle->setVertex( i, v );
			}

			// Scale the extrusion
			if ( triangle->isExtruded() )
			{
				v = triangle->getExtrusion();
				v.mul( sscale );
				triangle->setExtrusion( v );
			}

			doneShape.newShape = triangle;
			break;
		}

	case hkcdShapeType::BOX:
		{
			hkpBoxShape* box = static_cast< hkpBoxShape* >( shape );

			// Scale the half extents
			hkVector4 halfExtents;
			halfExtents = box->getHalfExtents();
			halfExtents.mul( sscale );
			box->setHalfExtents( halfExtents );

			doneShape.newShape = box;
			break;
		}

	case hkcdShapeType::CAPSULE:
		{
			hkpCapsuleShape* capsule = static_cast< hkpCapsuleShape* >( shape );

			// Scale the radius
			capsule->setRadius( sscale.getReal() * capsule->getRadius() );

			// Scale the vertices
			for ( int i = 0; i < 2; i++ )
			{
				hkVector4 vertex;
				vertex = capsule->getVertex( i );
				vertex.mul( sscale );
				capsule->setVertex( i, vertex );
			}
			
			doneShape.newShape = capsule;
			break;
		}

	case hkcdShapeType::CONVEX_VERTICES:
		{
			hkpConvexVerticesShape* cvs = static_cast< hkpConvexVerticesShape* >( shape );

			// Vertices
			{
				hkArray<hkVector4> vertices( cvs->getNumCollisionSpheres() );
				cvs->getOriginalVertices( vertices );

				const int numVertices = vertices.getSize();
				for ( int i = 0; i < numVertices; i++ )
				{
					vertices[ i ].mul( sscale );
				}

				cvs->copyVertexData( &vertices.begin()[0](0), sizeof(hkVector4), vertices.getSize() );
			}
			
			// Planes
			{
				hkVector4 convexOffset;
				hkVector4 planeScale;
				convexOffset.set( 0.0f, 0.0f, 0.0f, cvs->getRadius() );
				planeScale.setXYZ_W(hkVector4::getConstant<HK_QUADREAL_1>(), sscale );

				hkArray<hkVector4> planeEquations;
				planeEquations = cvs->getPlaneEquations();

				const int numPlaneEquations = planeEquations.getSize();
				for ( int i = 0; i < numPlaneEquations; i++ )
				{
					planeEquations[ i ].sub( convexOffset );
					planeEquations[ i ].mul( planeScale );
					planeEquations[ i ].add( convexOffset );
				}

				cvs->setPlaneEquations( planeEquations );
			}

			
			doneShape.newShape = cvs;
			break;
		}

	case hkcdShapeType::CONVEX_TRANSLATE:
		{
			hkpConvexTranslateShape* ts = static_cast< hkpConvexTranslateShape* >( shape );

			// Grab the child shape and scale it
			_scaleShape( const_cast< hkpConvexShape* >( ts->getChildShape() ), sscale, doneShapes );

			// Scale the translation
			ts->getTranslation().mul( sscale );

			// Update the convex radius
			ts->setRadius(ts->getChildShape()->getRadius());

			doneShape.newShape = ts;
			break;
		}

	case hkcdShapeType::CONVEX_TRANSFORM:
		{
			hkpConvexTransformShape* ts = static_cast< hkpConvexTransformShape* >( shape );

			// Grab the child shape and scale it
			_scaleShape( const_cast< hkpConvexShape* >( ts->getChildShape() ), sscale, doneShapes );

			// Scale the translation
			const hkQsTransform transform = ts->getQsTransform();
			hkVector4 translation = transform.getTranslation();
			translation.mul( sscale );
			hkQsTransform newTransform = transform;
			newTransform.setTranslation(translation);
			ts->setTransform(newTransform);

			// Update the convex radius
			ts->setRadius(ts->getChildShape()->getRadius());

			doneShape.newShape = ts;
			break;
		}

	case hkcdShapeType::TRANSFORM:
		{
			hkpTransformShape* ts = static_cast< hkpTransformShape* >( shape );

			// Grab the child shape and scale it
			_scaleShape( const_cast< hkpShape* >( ts->getChildShape() ), sscale, doneShapes );

			// Scale the transform
			hkTransform t = ts->getTransform();
			t.getTranslation().mul( sscale );
			ts->setTransform( t );

			doneShape.newShape = ts;
			break;
		}

	case hkcdShapeType::CONVEX_LIST:
	case hkcdShapeType::LIST:
	case hkcdShapeType::COLLECTION:
	case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION:
		{
			const hkpShapeContainer* shapeContainer = shape->getContainer();
			HK_ASSERT2( 0xDBC03771, shapeContainer, "Shape to be scaled returned a null shape container" );

			hkpShapeBuffer buffer;

			// For each shape in the collection
			for ( hkpShapeKey key = shapeContainer->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = shapeContainer->getNextKey( key ) )
			{
				const hkpShape* child = shapeContainer->getChildShape(key, buffer );

				// Check that a valid shape was returned, not a generated one
				HK_ASSERT2( 0x408e84c7, child != reinterpret_cast< hkpShape* >( &buffer ), "Abstract collections are not supported by hkpShapeScalingUtility" );

				_scaleShape( const_cast< hkpShape* >( child ), sscale, doneShapes );
			}
			
			// Calc AABB
			if( shape->getType() == hkcdShapeType::LIST )
			{
				hkpListShape* list = static_cast< hkpListShape* >( shape );
				list->recalcAabbExtents();
			}

			doneShape.newShape = shape;
			break;
		}

	case hkcdShapeType::EXTENDED_MESH:
		{
			hkpExtendedMeshShape* ems = static_cast< hkpExtendedMeshShape* >( shape );

			// Scale the triangle subparts
			for ( int i = 0; i < ems->getNumTrianglesSubparts(); ++i )
			{
				hkpExtendedMeshShape::TrianglesSubpart& part = ems->getTrianglesSubpartAt(i);
				hkVector4 scaling;
				scaling.setMul( sscale, part.getScaling() );
				part.setScaling( scaling );
				part.m_transform.m_translation.mul( sscale );
			}

			// Scale the shape subparts
			for (int i = 0; i < ems->getNumShapesSubparts(); i++)
			{
				hkpExtendedMeshShape::ShapesSubpart& part = ems->getShapesSubpartAt(i);
				const int numConvexChildren = part.m_childShapes.getSize();
				for (int k = 0; k < numConvexChildren; k++)
				{
					hkpConvexShape* convexChild = part.m_childShapes[k];
					_scaleShape(convexChild, sscale, doneShapes);
				}
			}

			// Calc AABB
			ems->recalcAabbExtents();

			doneShape.newShape = ems;
			break;
		}

	case hkcdShapeType::TRIANGLE_COLLECTION:
		{
			hkpSimpleMeshShape* sms = static_cast< hkpSimpleMeshShape* >( shape );

			// Scale all of the vertices
			const int n = sms->m_vertices.getSize();
			for ( int i = 0; i < n; i++ )
			{
				sms->m_vertices[ i ].mul( sscale );
			}

			doneShape.newShape = sms;
			break;
		}

	case hkcdShapeType::BV:
		{
			hkpBvShape* bvs = static_cast< hkpBvShape* >( shape );

			const hkpShape* child = bvs->getChildShape();
			const hkpShape* bv = bvs->getBoundingVolumeShape();

			_scaleShape( const_cast< hkpShape* >( child ), sscale, doneShapes );
			_scaleShape( const_cast< hkpShape* >( bv ), sscale, doneShapes );

			doneShape.newShape = bvs;
			break;
		}

	case hkcdShapeType::MOPP:
		{
			hkpMoppBvTreeShape* bvs = static_cast< hkpMoppBvTreeShape* >( shape );

			hkpShapeCollection* collection = const_cast< hkpShapeCollection* >( bvs->getShapeCollection() );
			_scaleShape( collection, sscale, doneShapes );

			hkpMoppCode* code = const_cast< hkpMoppCode* >( bvs->getMoppCode() );
			hkVector4 codeMult; codeMult.setAll(sscale);	codeMult.setComponent<3>(sscale.reciprocal());
			code->m_info.m_offset.mul(codeMult);

			bvs->m_codeInfoCopy = code->m_info.m_offset;

			doneShape.newShape = bvs;
			break;
		}

	default:
		{
			HK_ASSERT2( 0x408e84c7, false, "Shape not supported by hkpShapeScalingUtility" );
			break;
		}
	}
	
	if(doneShape.newShape != HK_NULL)
	{
		doneShapes.pushBack(doneShape);
	}

	return doneShape.newShape;
}


hkpShape* hkpShapeScalingUtility::scaleShapeSimd( hkpShape* shape, hkSimdRealParameter scale, hkArray<hkpShapeScalingUtility::ShapePair>* doneShapes )
{
	if (doneShapes)
	{
		return _scaleShape(shape, scale, *doneShapes);
	}
	else
	{
		// make a temp one
		hkArray<ShapePair> shapeCache;
		return _scaleShape(shape, scale, shapeCache);
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
