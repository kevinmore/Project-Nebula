/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Collide/ShapeUtils/CollapseTransform/hkpTransformCollapseUtil.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>

#include <Common/Internal/ConvexHull/hkGeometryUtility.h>

#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTransform/hkpConvexTransformShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>


hkpTransformCollapseUtil::Options::Options () : 
	m_sharedShapeBehaviour(ALWAYS_COLLAPSE), 
	m_sharedShapeThreshold(0), 
	m_propageTransformInList(true) 
{

}

hkpTransformCollapseUtil::Results::Results() : 
	m_numCollapsedShapes(0), 
	m_numIdentityTransformsRemoved(0), 
	m_numSpecializedTransformShapes(0), 
	m_numPropagatedTransformsToLists(0)
{
	reset();
}

void hkpTransformCollapseUtil::Results::reset()
{
	m_numCollapsedShapes = 0;
    m_numIdentityTransformsRemoved = 0;
    m_numSpecializedTransformShapes = 0;
    m_numPropagatedTransformsToLists =0;
}

void hkpTransformCollapseUtil::Results::report() const
{
	HK_REPORT("Number of collapsed shapes :"<<m_numCollapsedShapes);
	HK_REPORT("Number of identity transforms removed : "<<m_numIdentityTransformsRemoved);
	HK_REPORT("Number of specialized transform shapes : "<<m_numSpecializedTransformShapes);
	HK_REPORT("Number of propagated transforms to lists : "<<m_numPropagatedTransformsToLists);
}

static void _getShapeChildren (const hkpShape* shape, hkArray<const hkpShape*>& childrenOut)
{
	switch (shape->getType())
	{
		case hkcdShapeType::LIST:
			{
				const hkpListShape* listshape = static_cast<const hkpListShape*> (shape);
				for (int n=0; n<listshape->getNumChildShapes(); n++) 
				{
					const hkpShape* listChildShape = listshape->getChildShapeInl(n); 
					childrenOut.pushBack(listChildShape);
				}
				break;
			}

		case hkcdShapeType::TRANSFORM:
			{
				const hkpTransformShape* transformShape = static_cast<const hkpTransformShape*> (shape);
				childrenOut.pushBack(transformShape->getChildShape());
				break;
			}

		case hkcdShapeType::CONVEX_TRANSFORM:
			{
				const hkpConvexTransformShape* convexTransformShape = static_cast<const hkpConvexTransformShape*> (shape);
				childrenOut.pushBack(convexTransformShape->getChildShape());
				break;
			}

		case hkcdShapeType::CONVEX_TRANSLATE:
			{
				const hkpConvexTranslateShape* convexTranslateShape = static_cast<const hkpConvexTranslateShape*> (shape);
				childrenOut.pushBack(convexTranslateShape->getChildShape());
				break;
			}
		default:
			break;
	}

}

class _SharedShapeData
{
	public:

		void removeSingleReferencedShapeData()
		{
			for (int i=m_sharedShapes.getSize()-1; i>=0; --i)
			{
				if (m_sharedShapes[i].m_numReferences<2)
				{
					if (m_sharedShapes[i].m_originalShape)
					{
						m_sharedShapes[i].m_originalShape->removeReference();
					}
					if (m_sharedShapes[i].m_replacementShape)
					{
						m_sharedShapes[i].m_replacementShape->removeReference();
					}
					m_sharedShapes.removeAt(i);
				}
			}
		}

		_SharedShapeData (const hkArray<hkpRigidBody*>& rigidBodies)
		{
			for (int i=0; i<rigidBodies.getSize(); i++)
			{
				recursivelyParseShapeReferences (rigidBodies[i]->getCollidable()->getShape());
			}

			removeSingleReferencedShapeData();
		}

		_SharedShapeData (const hkpShape* shape)
		{
			recursivelyParseShapeReferences (shape);
			removeSingleReferencedShapeData();
		}

		~_SharedShapeData ()
		{
			for (int i=0; i<m_sharedShapes.getSize(); i++)
			{
				if (m_sharedShapes[i].m_originalShape)
				{
					m_sharedShapes[i].m_originalShape->removeReference();
				}
				if (m_sharedShapes[i].m_replacementShape)
				{
					m_sharedShapes[i].m_replacementShape->removeReference();
				}
			}
		}

		const hkpShape* findReplacement (const hkpShape* original) const
		{
			for (int i=0; i<m_sharedShapes.getSize(); i++)
			{
				if (m_sharedShapes[i].m_originalShape == original)
				{
					return m_sharedShapes[i].m_replacementShape;
				}
			}

			return HK_NULL;

		}

		int numberOfReferences (const hkpShape* shape) const
		{
			for (int i=0; i<m_sharedShapes.getSize(); i++)
			{
				if ( (m_sharedShapes[i].m_originalShape == shape) ||
					 (m_sharedShapes[i].m_replacementShape == shape) )
				{
					return m_sharedShapes[i].m_numReferences;
				}
			}

			return 0;
		}

		hkResult setReplacement(const hkpShape* shape, const hkpShape* replacement)
		{
			for (int i=0; i<m_sharedShapes.getSize(); i++)
			{
				if ( (m_sharedShapes[i].m_originalShape == shape) ||
					(m_sharedShapes[i].m_replacementShape == shape) )
				{
					replacement->addReference();

					if (m_sharedShapes[i].m_replacementShape)
					{
						m_sharedShapes[i].m_replacementShape->removeReference();
					}

					m_sharedShapes[i].m_replacementShape = replacement;
					return HK_SUCCESS;
				}
			}

			return HK_FAILURE;
		}

	private:

		struct SharedShape
		{
			const hkpShape* m_originalShape;
			const hkpShape* m_replacementShape;
			int m_numReferences;
		};

		hkArray<SharedShape> m_sharedShapes;

		int findShapeDataIndex (const hkpShape* shape) const
		{
			for (int i=0; i<m_sharedShapes.getSize(); i++)
			{
				if (m_sharedShapes[i].m_originalShape == shape)
				{
					return i;
				}
			}
			return -1;
		}


		void recursivelyParseShapeReferences (const hkpShape* shape);
};


void _SharedShapeData::recursivelyParseShapeReferences (const hkpShape* shape)
{
	int index = findShapeDataIndex(shape);

	if (index>=0)
	{
		m_sharedShapes[index].m_numReferences++;
		// we don't recurse once we find a shared shape as somebody else already has
		return;
	}
	else
	{
		SharedShape sharedShape;
		sharedShape.m_originalShape = shape;
		sharedShape.m_replacementShape = HK_NULL;
		sharedShape.m_numReferences = 1;
		m_sharedShapes.pushBack(sharedShape);
		shape->addReference();
	}

	hkArray<const hkpShape*> children;
	_getShapeChildren(shape, children);

	for (int i=0; i<children.getSize(); i++)
	{
		recursivelyParseShapeReferences(children[i]);
	}
}


/*static*/ hkResult hkpTransformCollapseUtil::collapseTransforms (hkpRigidBody* rigidBody, const Options& options, Results& resultsOut)
{
	hkArray<hkpRigidBody*> tempArray;
	tempArray.pushBack(rigidBody);

	return collapseTransforms(tempArray, options, resultsOut);
}


/*static*/ hkResult hkpTransformCollapseUtil::collapseTransforms (hkArray<hkpRigidBody*>& rigidBodies, const Options& options, Results& resultsOut)
{
	resultsOut.reset();

	_SharedShapeData sharedShapeData (rigidBodies);

	for (int i=0; i<rigidBodies.getSize(); i++)
	{
		hkpRigidBody* rb = rigidBodies[i];
		const hkpShape* originalShape = rb->getCollidable()->getShape();
		const hkpShape* newShape = collapseShapesRecursively (originalShape, options, sharedShapeData, resultsOut);

		//remove the previous shape
		originalShape->removeReference();

		rb->getCollidableRw()->setShape(const_cast<hkpShape*>(newShape));

	}

	return HK_SUCCESS;
}

/*static*/ hkResult hkpTransformCollapseUtil::collapseTransforms(const hkpShape* shape, const Options& options, Results& resultsOut, hkpShape** shapeOut)
{
	_SharedShapeData sharedShapeData(shape);

	const hkpShape* newShape = collapseTransformShape(shape, options, sharedShapeData, resultsOut);
	*shapeOut = const_cast<hkpShape*>(newShape);

	return HK_SUCCESS;
}

/*static*/ const hkpShape* hkpTransformCollapseUtil::collapseShapesRecursively (const hkpShape* shape, const Options& options, class _SharedShapeData& sharedShapeData, Results& resultsOut)
{
	const hkpShape* replacement = sharedShapeData.findReplacement(shape);
	if (replacement)
	{
		replacement->addReference();
		return replacement;
	}

	// else

	switch (shape->getType())
	{
		case hkcdShapeType::LIST:
			{
				// TODO : handle single child list shapes

				const hkpListShape* listShape = static_cast<const hkpListShape*> (shape);
				int nChildren = listShape->getNumChildShapes();

				hkArray<const hkpShape*> newShapes;

				bool childrenUnmodified = true;
				for (int n=0; n<nChildren; n++) 
				{
					const hkpShape* childShape = listShape->getChildShapeInl(n); 
					const hkpShape* newShape = collapseShapesRecursively(childShape, options, sharedShapeData, resultsOut);		
					newShapes.pushBack(newShape);

					if (newShape != childShape) childrenUnmodified = false;
				}

				//if none of the children were modified, remove a reference from each child shape
				//and return the original list plus a reference
				if (childrenUnmodified) 
				{
					for (int n=0; n<nChildren; n++) 
					{
						newShapes[n]->removeReference();
					}
					listShape->addReference();
					return listShape;
				}

				//otherwise create a new list, and relinquish ownership of the newly created children to the new list
				hkpListShape* newListShape = new hkpListShape(newShapes.begin(), nChildren);
				for (int n=0; n<nChildren; n++) 
				{
					newShapes[n]->removeReference();		
				}
				
				sharedShapeData.setReplacement(listShape, newListShape);

				replacement = newListShape;
				break;
			}

		case hkcdShapeType::TRANSFORM:
			{
				const hkpTransformShape* transformShape = static_cast<const hkpTransformShape*> (shape);
				const hkpShape* originalChild = transformShape->getChildShape();
				const hkpShape* newChild = collapseShapesRecursively(originalChild, options, sharedShapeData, resultsOut);

				const hkpTransformShape* newTransformShape;
				{
					if (originalChild==newChild)
					{
						newChild->removeReference();
						newTransformShape = transformShape;
						transformShape->addReference();
					}
					else
					{
						newTransformShape = new hkpTransformShape (newChild, transformShape->getTransform());
						newChild->removeReference();
						sharedShapeData.setReplacement(transformShape, newTransformShape);
					}					
				}

				// The child has been collapsed, we now need to possibly collapse this guy into the child
				const hkpShape* collapsedShape = collapseTransformShape (newTransformShape, options, sharedShapeData, resultsOut);
				newTransformShape->removeReference();

				replacement = collapsedShape;
				break;
			}

		case hkcdShapeType::CONVEX_TRANSFORM:
			{
				const hkpConvexTransformShape* convexTransformShape = static_cast<const hkpConvexTransformShape*> (shape);
				const hkpConvexShape* originalChild = convexTransformShape->getChildShape();
				const hkpConvexShape* newChild = static_cast<const hkpConvexShape*> (collapseShapesRecursively(originalChild, options, sharedShapeData, resultsOut));

				const hkpConvexTransformShape* newConvexTransformShape;
				{
					if (originalChild==newChild)
					{
						newChild->removeReference();
						newConvexTransformShape = convexTransformShape;
						convexTransformShape->addReference();
					}
					else
					{
						hkTransform transform; convexTransformShape->getTransform( &transform );
						newConvexTransformShape = new hkpConvexTransformShape (newChild, transform );
						newChild->removeReference();
						sharedShapeData.setReplacement(convexTransformShape, newConvexTransformShape);
					}					
				}

				// The child has been collapsed, we now need to possibly collapse this guy into the child
				const hkpShape* collapsedShape = collapseTransformShape (newConvexTransformShape, options, sharedShapeData, resultsOut);
				newConvexTransformShape->removeReference();
				
				replacement = collapsedShape;
				break;
			}

		case hkcdShapeType::CONVEX_TRANSLATE:
			{
				const hkpConvexTranslateShape* convexTranslateShape = static_cast<const hkpConvexTranslateShape*> (shape);
				const hkpConvexShape* originalChild = convexTranslateShape->getChildShape();
				const hkpConvexShape* newChild = static_cast<const hkpConvexShape*> (collapseShapesRecursively(originalChild, options, sharedShapeData, resultsOut));

				const hkpConvexTranslateShape* newConvexTranslateShape;
				{
					if (originalChild==newChild)
					{
						newChild->removeReference();
						newConvexTranslateShape = convexTranslateShape;
						convexTranslateShape->addReference();
					}
					else
					{
						newConvexTranslateShape = new hkpConvexTranslateShape (newChild, convexTranslateShape->getTranslation());
						newChild->removeReference();
						sharedShapeData.setReplacement(convexTranslateShape, newConvexTranslateShape);
					}					
				}

				// The child has been collapsed, we now need to possibly collapse this guy into the child
				const hkpShape* collapsedShape = collapseTransformShape (newConvexTranslateShape, options, sharedShapeData, resultsOut);
				newConvexTranslateShape->removeReference();

				replacement = collapsedShape;
				break;
			}

		default:
			{
				// Nothing to do, pretend we created a new shape
				shape->addReference();
				replacement = shape;
			}
	}
	return replacement;
}

/*static*/ const hkpShape* hkpTransformCollapseUtil::transformTransformShape (const hkpShape* shape, const hkTransform& transform )
{
	if ( transform.isApproximatelyEqual( hkTransform::getIdentity() ))
	{
		shape->addReference();
		return shape;
	}
	hkpShapeType transformShapeType = shape->getType();
	{	// translation only
		switch ( transformShapeType )
		{
		case hkcdShapeType::TRANSFORM:
			{
				const hkpTransformShape* tformshape = (const hkpTransformShape*) (shape);
				hkTransform t;	t.setMul( transform, tformshape->getTransform());

				if ( t.isApproximatelyEqual( hkTransform::getIdentity() ))
				{
					const hkpShape* childShape = tformshape->getChildShape();
					childShape->addReference();
					return childShape;
				}
				return new hkpTransformShape( tformshape->getChildShape(), t );
			} 
		case hkcdShapeType::CONVEX_TRANSFORM:
			{
				const hkpConvexTransformShape* tformshape = (const hkpConvexTransformShape*) (shape);
				hkTransform localTransform; tformshape->getTransform( &localTransform );
				hkTransform t;	t.setMul( transform, localTransform );

				if ( t.isApproximatelyEqual( hkTransform::getIdentity() ))
				{
					const hkpShape* childShape = tformshape->getChildShape();
					childShape->addReference();
					return childShape;
				}
				return new hkpConvexTransformShape( tformshape->getChildShape(), t );
			}
		case hkcdShapeType::CONVEX_TRANSLATE:
			if ( transform.getRotation().isApproximatelyEqual( hkTransform::getIdentity().getRotation() ))
			{
				const hkpConvexTranslateShape* ctlateshape = (const hkpConvexTranslateShape*) (shape);
				hkTransform tr; tr.setIdentity();
				tr.setTranslation( ctlateshape->getTranslation() );
				hkTransform t;	t.setMul( transform, tr);

				if ( t.isApproximatelyEqual( hkTransform::getIdentity() ))
				{
					const hkpShape* childShape = ctlateshape->getChildShape();
					childShape->addReference();
					return childShape;
				}
				return new hkpConvexTranslateShape( ctlateshape->getChildShape(), t.getTranslation() );
			}
		default:
			if ( shape->isConvex() )
			{
				const hkpConvexShape* convexShape = static_cast<const hkpConvexShape*>( shape );
				hkpConvexTransformShape* newT = new hkpConvexTransformShape( convexShape, transform );
				return newT;
			}
			else
			{
				hkpTransformShape* newT = new hkpTransformShape( shape, transform );
				return newT;
			}
		}
	}
}

/*static*/ const hkpShape* hkpTransformCollapseUtil::collapseTransformShape (const hkpShape* transformShape, const Options& options, class _SharedShapeData& sharedShapeData, Results& resultsOut)
{
	hkpShapeType transformShapeType = transformShape->getType();
	hkTransform parentFromChild;
	
	const hkpShape* childShape = HK_NULL;

	switch ( transformShapeType )
	{
		case hkcdShapeType::TRANSFORM:
			{
				const hkpTransformShape* tformshape = static_cast<const hkpTransformShape*> (transformShape);
				parentFromChild = tformshape->getTransform();
				childShape = tformshape->getChildShape();
				break;
			}
		case hkcdShapeType::CONVEX_TRANSFORM:
			{
				const hkpConvexTransformShape* ctformshape = static_cast<const hkpConvexTransformShape*> (transformShape);
				ctformshape->getTransform( &parentFromChild );
				childShape = ctformshape->getChildShape();
				break;
			}
		case hkcdShapeType::CONVEX_TRANSLATE:
			{
				const hkpConvexTranslateShape* ctlateshape = static_cast<const hkpConvexTranslateShape*> (transformShape);
				parentFromChild.setIdentity();
				parentFromChild.setTranslation( ctlateshape->getTranslation() );
				childShape = ctlateshape->getChildShape();
				break;
			}
		default:
			// Shouldn't get here
			transformShape->addReference();
			return transformShape;
	}

	// Otherwise : do different stuff depending on the shape type
	const hkpShapeType childShapeType = childShape->getType();

	// If the child is a sphere, ignore any rotations
	if (childShapeType == hkcdShapeType::SPHERE)
	{
		parentFromChild.setRotation(hkTransform::getIdentity().getRotation());
	}

	// First case : transform is identity -> return child shape
	if (parentFromChild.isApproximatelyEqual(hkTransform::getIdentity()))
	{
		resultsOut.m_numIdentityTransformsRemoved++;
		childShape->addReference();
		sharedShapeData.setReplacement(transformShape, childShape);
		return childShape;
	}

	// Possibly do not collapse
	bool allowChangesToChildShape = true;
	{
		switch (options.m_sharedShapeBehaviour)
		{
			case ALWAYS_COLLAPSE:
				allowChangesToChildShape = true;
				break;
			case NEVER_COLLAPSE:
				allowChangesToChildShape = false;
				break;
			case COLLAPSE_IF_LESS_THAN_THRESHOLD:
				allowChangesToChildShape = sharedShapeData.numberOfReferences(childShape)<options.m_sharedShapeThreshold;
				break;
		}
	}

	// Handle list shape case (T->L->x,y,z..)
	if (childShapeType==hkcdShapeType::LIST)
	{
		const hkpListShape* listShape = static_cast<const hkpListShape*> (childShape);
		if (!options.m_propageTransformInList && allowChangesToChildShape)
		{
			resultsOut.m_numPropagatedTransformsToLists++;

			// Create L'->(T1->x),(T2->y),(T3->z)
			hkArray<const hkpShape*> newChildren;
			for (int i=0; i<listShape->getNumChildShapes(); i++)
			{
				const hkpShape* grandChildShape = listShape->getChildShapeInl(i);
				hkpTransformShape* childTransformShape = new hkpTransformShape (grandChildShape, parentFromChild);
				const hkpShape* collapsedGrandChild = collapseTransformShape(childTransformShape, options, sharedShapeData, resultsOut);
				newChildren.pushBack(collapsedGrandChild);
				childTransformShape->removeReference();				
			}

			const hkpListShape* newListShape = new hkpListShape(newChildren.begin(), newChildren.getSize());
			for (int n=0; n<newChildren.getSize(); n++) 
			{
				newChildren[n]->removeReference();		
			}

			sharedShapeData.setReplacement(transformShape, newListShape);
			return newListShape;
		}
		else
		{
			transformShape->addReference();
			return transformShape;
		}
	}

	// Possibly, collapse into the child shape
	if (allowChangesToChildShape)
	{
		switch (childShapeType)
		{
			case hkcdShapeType::CAPSULE:
				{
					// Capsule : Transform the vertices of the capsule (same as cylinder)
					const hkpCapsuleShape* childCapsuleShape = static_cast<const hkpCapsuleShape*> (childShape);

					const hkVector4& childA = childCapsuleShape->getVertices()[0];
					const hkVector4& childB = childCapsuleShape->getVertices()[1];
					const hkReal radius = childCapsuleShape->getRadius();

					hkVector4 parentA;
					parentA.setTransformedPos ( parentFromChild, childA );
					hkVector4 parentB;
					parentB.setTransformedPos ( parentFromChild, childB	);

					const hkpCapsuleShape* newCapsule = new hkpCapsuleShape(parentA, parentB, radius);

					sharedShapeData.setReplacement(transformShape, newCapsule);

					resultsOut.m_numCollapsedShapes++;
					return newCapsule;
				}

			case hkcdShapeType::CYLINDER:
				{
					// Cylinder : Transform both vertices (same as capsule)
					const hkpCylinderShape* childCylinderShape = static_cast<const hkpCylinderShape*> (childShape);

					const hkVector4& childA = childCylinderShape->getVertices()[0];
					const hkVector4& childB = childCylinderShape->getVertices()[1];
					const hkReal cylRadius = childCylinderShape->getCylinderRadius();
					const hkReal radius = childCylinderShape->getRadius();

					hkVector4 parentA; 
					parentA.setTransformedPos ( parentFromChild, childA );
					hkVector4 parentB;
					parentB.setTransformedPos ( parentFromChild, childB	);

					const hkpCylinderShape* newCylinder = new hkpCylinderShape(parentA, parentB, cylRadius, radius);

					sharedShapeData.setReplacement(transformShape, newCylinder);

					resultsOut.m_numCollapsedShapes++;
					return newCylinder;
				}

			case hkcdShapeType::CONVEX_VERTICES:
				{
					// Convex Vertices : Transform the vertices and planes
					const hkpConvexVerticesShape* childCVShape = static_cast<const hkpConvexVerticesShape*> (childShape);

					//transform vertices
					hkArray<hkVector4> vertices;
					childCVShape->getOriginalVertices(vertices);
					hkVector4Util::transformPoints(parentFromChild, vertices.begin(), vertices.getSize(), vertices.begin());
					hkStridedVertices newverts;
					newverts.m_numVertices = vertices.getSize();
					newverts.m_striding = sizeof(hkVector4);
					newverts.m_vertices = &(vertices[0](0));

					//transform plane equations
					const hkArray<hkVector4>& planes = childCVShape->getPlaneEquations();
					hkArray<hkVector4> newplanes; newplanes.setSize(planes.getSize());
					hkVector4 pivotShift = parentFromChild.getTranslation();
					for (int p = 0; p < planes.getSize(); p++)
					{
						hkVector4 plane = planes[p];
						hkSimdReal origDist = plane.getW();
						plane.setRotatedDir(parentFromChild.getRotation(), plane);
						hkSimdReal newDist = origDist - pivotShift.dot<3>( plane );
						plane.setW(newDist);
						newplanes[p] = plane;
					}

					const hkpConvexVerticesShape* newCVShape = new hkpConvexVerticesShape(newverts, newplanes, childCVShape->getRadius());

					sharedShapeData.setReplacement(transformShape, newCVShape);

					resultsOut.m_numCollapsedShapes++;
					return newCVShape;
				}

			case hkcdShapeType::BOX:
				{
					// Box : if transform consists of 90 degree rotations (plus any translation),
					//       collapse by returning a box with modified half-extents, 
					//       wrapped in an hkpConvexTranslateShape if the translation is non-zero.
					
					//check if the rotation matrix is valid first
					if ( !(parentFromChild.getRotation().isOrthonormal() && parentFromChild.getRotation().isOk()) )
					{
						break;
					}

					//check if transform is just a translation, in which case the extents won't be modified
					if ( parentFromChild.getRotation().isApproximatelyEqual(hkTransform::getIdentity().getRotation()) )
					{
						break;
					}

					const hkSimdReal tol = hkSimdReal::fromFloat(1.0e-4f);
					int map[3];

					// If the rotated coordinate system lines up with the axes, we can replace the transform with modified box extents
					// (otherwise just proceed below to try to replace transform type with convex transform or translate).
					
					bool failedA = false;
					bool failedB = false;

					for (int i=0; i<3; i++)
					{
						//columns of the rotation matrix are the transformed basis vectors
						hkVector4& axisRot = parentFromChild.getRotation().getColumn(i);

						//rotation takes box to box iff a) && b), where:

						// a) each element of each column has magnitude zero or one, within tolerance
						bool passedA1 = false;

						hkVector4 absRot; absRot.setAbs(axisRot);
						for (int j=0; j<3; j++)
						{ 
							//to pass, at least one of these must be true
							hkSimdReal absRotj = absRot.getComponent(j); 
							hkBool32 testA0 = absRotj.isLess(tol);
							
							absRotj.sub(hkSimdReal_1);
							absRotj.setAbs(absRotj);
							hkBool32 testA1 = absRotj.isLess(tol);
							
							if (!(testA0 | testA1)) 
					 		{
								failedA = true;
								break;
							}

							if (testA1) 
							{
								//jth element is 1, thus the new j-axis corresponds to the old i-axis 
								map[j] = i;
								passedA1 = true;
							}
						}
						if (!passedA1) failedA = true; //one of the elements must be 1.0, within tolerance
							
						if (failedA) break; 
						
						// b) the sum of the magnitudes of the elements in each column is one, within tolerance 
						const hkSimdReal sum = absRot.horizontalAdd<3>() - hkSimdReal_1;
						failedB = !( sum.isLess(tol) );
						
						if (failedB) break;
					}
		
					if (failedA || failedB) break; //proceed to switch below

					//transform is of right type, now modify extents
					const hkpBoxShape* childBoxShape = static_cast<const hkpBoxShape*> (childShape);
					const hkVector4& halfExtents = childBoxShape->getHalfExtents();
					hkVector4 newHalfExtents; newHalfExtents.set( halfExtents(map[0]), halfExtents(map[1]), halfExtents(map[2]) );
					
					const hkpBoxShape* newBoxShape = new hkpBoxShape( newHalfExtents );

					// if translation is zero, just return the new box shape
					if ( parentFromChild.getTranslation().lengthSquared<3>() < (tol*tol) )
					{						
						sharedShapeData.setReplacement(transformShape, newBoxShape);
						resultsOut.m_numCollapsedShapes++;
						return newBoxShape;
					}

					// otherwise, wrap new box in an hkpConvexTranslateShape
					const hkpConvexTranslateShape* newShape = new hkpConvexTranslateShape(newBoxShape, parentFromChild.getTranslation());
					newBoxShape->removeReference();
					sharedShapeData.setReplacement(transformShape, newShape);
					resultsOut.m_numSpecializedTransformShapes++;
					return newShape;
				}
			default:
				break;
		}
	}

	// else (do not collapse, but possibly replace transform type)
	switch (childShapeType)
	{
		case hkcdShapeType::CAPSULE:
		case hkcdShapeType::CYLINDER:
		case hkcdShapeType::CONVEX_VERTICES:
		case hkcdShapeType::CONVEX:
		case hkcdShapeType::SPHERE:
		case hkcdShapeType::BOX:
			{
				// Convex shapes : Use hkConvexTranslate or hkConvexTransform shapes
				const hkpConvexShape* childConvexShape = static_cast<const hkpConvexShape*> (childShape);

				// Is is just translation? (sphere's rotation can safely be ignored)
				const bool translateOnly = parentFromChild.getRotation().isApproximatelyEqual(hkTransform::getIdentity().getRotation());

				if (translateOnly && (transformShapeType != hkcdShapeType::CONVEX_TRANSLATE))
				{
					const hkpConvexTranslateShape* newShape = new hkpConvexTranslateShape(childConvexShape, parentFromChild.getTranslation());
					
					sharedShapeData.setReplacement(transformShape, newShape);

					resultsOut.m_numSpecializedTransformShapes++;
					return newShape;
				}
				else if (transformShapeType == hkcdShapeType::TRANSFORM)
				{
					const hkpConvexTransformShape* newShape = new hkpConvexTransformShape(childConvexShape, parentFromChild);

					sharedShapeData.setReplacement(transformShape, newShape);

					resultsOut.m_numSpecializedTransformShapes++;
					return newShape;
				}

				break;
			}
		default:
			break;
	}

	// We couldn't collapse anything, return the same shape
	transformShape->addReference();
	return transformShape;
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
