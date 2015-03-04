/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

// Needed for the class reflection
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkTypeInfo.h>



// this
#include <Common/GeometryUtilities/Mesh/Default/hkDefaultCompoundMeshShape.h>

HK_REFLECTION_DEFINE_STUB_VIRTUAL(hkDefaultCompoundMeshShape, hkMeshShape);
HK_REFLECTION_DEFINE_STUB_VIRTUAL(hkDefaultCompoundMeshBody, hkMeshBody);

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                     hkDefaultCompoundMeshShape

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

hkDefaultCompoundMeshShape::hkDefaultCompoundMeshShape(const hkMeshShape*const* shapes, const hkMatrix4* transforms, int numShapes)
{
    m_shapes.setSize(numShapes);
    for (int i = 0; i < numShapes; i++)
    {
        const hkMeshShape* shape = shapes[i];
        if (shape)
        {
            shape->addReference();
            int numSections = shape->getNumSections();

            for (int j = 0; j < numSections; j++)
            {
                MeshSection& section = m_sections.expandOne();
                section.m_shapeIndex = i;
                section.m_sectionIndex = j;
            }
        }
        m_shapes[i] = shape;
    }
	if ( transforms )
	{
		m_defaultChildTransforms.reserve( numShapes );
		m_defaultChildTransforms.insertAt( 0, transforms, numShapes );
	}
}

hkDefaultCompoundMeshShape::~hkDefaultCompoundMeshShape()
{
    const int numShapes = m_shapes.getSize();
    for (int i = 0; i < numShapes; i++)
    {
        const hkMeshShape* shape = m_shapes[i];
        if (shape)
        {
            shape->removeReference();
        }
    }
}

int hkDefaultCompoundMeshShape::getNumSections() const
{
    return m_sections.getSize();
}

void hkDefaultCompoundMeshShape::lockSection(int sectionIndex, hkUint8 accessFlags, hkMeshSection& sectionOut) const
{
    const MeshSection& section = m_sections[sectionIndex];
    const hkMeshShape* shape = m_shapes[section.m_shapeIndex];

    shape->lockSection(section.m_sectionIndex, accessFlags, sectionOut);


	/// It shouldnt' have a transform index... 
	HK_ASSERT(0x3424305, sectionOut.m_transformIndex == -1);

    sectionOut.m_transformIndex = section.m_shapeIndex;
	sectionOut.m_sectionIndex = sectionIndex;
}

void hkDefaultCompoundMeshShape::unlockSection(const hkMeshSection& lockedSection) const
{
	const MeshSection& section = m_sections[lockedSection.m_sectionIndex];
    const hkMeshShape* shape = m_shapes[section.m_shapeIndex];

	hkMeshSection internalSection = lockedSection;
	internalSection.m_transformIndex = -1;
	internalSection.m_sectionIndex = section.m_sectionIndex;

	shape->unlockSection(internalSection);
}

/* static */hkBool hkDefaultCompoundMeshShape::hasIndexTransforms(const hkMeshSectionCinfo* sections, int numSections)
{
	const hkMeshSectionCinfo* end = sections + numSections;
	if ( numSections <=0 )
	{
		return false;
	}

    for (; sections != end; sections ++ )
    {
        if (sections->m_transformIndex < 0)
        {
            return false;
        }
    }

    return true;
}


static HK_FORCE_INLINE bool hkStandardCompoundShape_orderSections(const hkMeshSection* a, const hkMeshSection* b)
{
    return a->m_transformIndex < b->m_transformIndex;
}

/* static */void hkDefaultCompoundMeshShape::createTransformIndexedShapeList(hkMeshSystem* meshSystem,  const hkMeshSectionCinfo* sectionsIn, int numSections, hkArray<const hkMeshShape*>& shapesOut)
{
	if ( numSections == 0 )
	{
		shapesOut.clear();
		return;
	}

    hkLocalBuffer<const hkMeshSectionCinfo*> sections(numSections);
    for (int i = 0; i < numSections; i++)
    {
        sections[i] = sectionsIn + i;
    }

    // Lets sort the sections by the transform index
    int numTransforms = sections[numSections - 1]->m_transformIndex + 1;

    // Create memory to hold the compound shapes pointers
    shapesOut.setSize(numTransforms);
    hkString::memSet(shapesOut.begin(), 0, sizeof(const hkMeshShape*) * numTransforms);

    // Work space for the sections
    hkLocalArray<hkMeshSectionCinfo> dstSections(numSections);
    {
        const hkMeshSectionCinfo** start = sections.begin();
        const hkMeshSectionCinfo** end = start + numSections;

        while (start < end)
        {
            const hkMeshSectionCinfo** cur = start + 1;

            int transformIndex = (*start)->m_transformIndex;
            while (cur < end && (*cur)->m_transformIndex == transformIndex)
            {
                cur++;
            }

            int numSubSections = int(cur - start);

            dstSections.setSize(numSubSections);
            for (int i = 0; i < numSubSections; i++)
            {
                // Copy the section
                dstSections[i] = *(start[i]);
                // We don't have any indexing
                dstSections[i].m_transformIndex = -1;
            }

            hkMeshShape* shape = meshSystem->createShape(dstSections.begin(), numSubSections );
            shapesOut[transformIndex] = shape;

            start = cur;
        }
    }
}

/* static */hkDefaultCompoundMeshShape* hkDefaultCompoundMeshShape::createTransformIndexedShape(hkMeshSystem* meshSystem,  const hkMeshSectionCinfo* sectionsIn, int numSections)
{
    // Holds the shapes indexed by transform. The size is just a guess - it could be larger.
    hkLocalArray<const hkMeshShape*> shapes(numSections);

    createTransformIndexedShapeList(meshSystem, sectionsIn, numSections, shapes);
    hkDefaultCompoundMeshShape* meshShape = new hkDefaultCompoundMeshShape(shapes.begin(), HK_NULL, shapes.getSize());
    for (int i = 0; i < shapes.getSize(); i++)
    {
        const hkMeshShape* shape = shapes[i];
        if (shape)
		{
			shape->removeReference();
		}
    }
    return meshShape;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                     hkDefaultCompoundMeshBody

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

hkDefaultCompoundMeshBody::hkDefaultCompoundMeshBody(hkMeshSystem* meshSystem, const hkDefaultCompoundMeshShape* shape, const hkMatrix4& transform, hkIndexedTransformSetCinfo* transformSetCinfo)
{
	// Save the transform
	m_transform = transform;

    if (transformSetCinfo)
    {
        m_transformSet = new hkIndexedTransformSet(*transformSetCinfo);
        m_transformSet->removeReference();
		if ( shape->m_defaultChildTransforms.getSize() == transformSetCinfo->m_numMatrices )
		{
			m_transformSet->setMatrices( 0, shape->m_defaultChildTransforms.begin(), shape->m_defaultChildTransforms.getSize() );
		}
    }
	else if ( !shape->m_defaultChildTransforms.isEmpty() )
	{
		hkIndexedTransformSetCinfo cinfo;
		cinfo.m_inverseMatrices = HK_NULL;
		cinfo.m_matrices = shape->m_defaultChildTransforms.begin();
		cinfo.m_numMatrices = shape->m_defaultChildTransforms.getSize();
		cinfo.m_allTransformsAreAffine = false;
		m_transformSet = new hkIndexedTransformSet(cinfo);
		m_transformSet->removeReference();
	}		



	//

    m_shape = shape;
    shape->addReference();

    const int numShapes = shape->m_shapes.getSize();

    m_bodies.setSize(numShapes);

    for (int i = 0; i < numShapes; i++)
    {
        const hkMeshShape* childShape = shape->m_shapes[i];
		if (childShape)
		{
			hkMeshBody* childBody = meshSystem->createBody(childShape, transform, HK_NULL);
			HK_ASSERT(0xa02f12, childBody != HK_NULL);
			m_bodies[i] = childBody;
		}
		else
		{
			m_bodies[i] = HK_NULL;
		}
    }

    // Set up initially
    m_transformIsDirty = true;
    m_transformSetUpdated = true;
    completeUpdate();
}

hkDefaultCompoundMeshBody::~hkDefaultCompoundMeshBody()
{
    m_shape->removeReference();
    const int numBodies = m_bodies.getSize();
    for (int i = 0; i < numBodies; i++)
    {
		hkMeshBody* body = m_bodies[i];
		if ( body )
		{
			body->removeReference();
		}
    }
}

const hkMeshShape* hkDefaultCompoundMeshBody::getMeshShape() const
{
    return m_shape;
}

void hkDefaultCompoundMeshBody::setTransform( const hkMatrix4& transform )
{
    m_transform = transform;
    m_transformIsDirty = true;
}

void hkDefaultCompoundMeshBody::getTransform( hkMatrix4& transform ) const
{
    transform = m_transform;
}

hkResult hkDefaultCompoundMeshBody::setPickingData(int id, void* data)
{
	const int numBodies = m_bodies.getSize();
	for (int i = 0; i < numBodies; i++)
	{
        hkMeshBody* body = m_bodies[i];
        if (body)
        {
            hkResult res = body->setPickingData(id, data);
            if (res != HK_SUCCESS)
            {
                return res;
            }
        }
	}
    return HK_SUCCESS;
}

hkMeshVertexBuffer* hkDefaultCompoundMeshBody::getVertexBuffer(int sectionIndex)
{
    const hkDefaultCompoundMeshShape::MeshSection& section = m_shape->m_sections[sectionIndex];
    hkMeshBody* body = m_bodies[section.m_shapeIndex];
    return body->getVertexBuffer(section.m_sectionIndex);
}

void hkDefaultCompoundMeshBody::completeUpdate()
{
    const int numBodies = m_bodies.getSize();

    if (m_transformSet)
    {
        if (m_transformIsDirty || m_transformSetUpdated)
        {
            hkLocalArray<hkMatrix4> matrices(m_bodies.getSize());
			m_transformSet->calculateMatrices(m_transform, matrices);

            for (int i = 0; i < numBodies; i++)
            {
                hkMeshBody* body = m_bodies[i];
                if (body)
                {
                    // Set the transform
					body->setTransform(matrices[i]);
                    body->completeUpdate();
                }
            }
            
            m_transformIsDirty = false;
            m_transformSetUpdated = false;
        }
    }
    else
    {
        for (int i = 0; i < numBodies; i++)
        {
            hkMeshBody* body = m_bodies[i];
            if (body)
            {
                body->setTransform(m_transform);
				body->completeUpdate();
            }
        }
		m_transformIsDirty = false;
    }
}

void hkDefaultCompoundMeshBody::completeUpdate(const hkMatrix4& transform)
{
	const int numBodies = m_bodies.getSize();

	if (m_transformSet)
	{
		hkLocalArray<hkMatrix4> matrices(m_bodies.getSize());
		m_transformSet->calculateMatrices(transform, matrices);

		for (int i = 0; i < numBodies; i++)
		{
			hkMeshBody* body = m_bodies[i];
			if (body)
			{
				// Set the transform
				body->completeUpdate(matrices[i]);
			}
		}

		m_transformSetUpdated = false;
	}
	else
	{
		for (int i = 0; i < numBodies; i++)
		{
			hkMeshBody* body = m_bodies[i];
			if (body)
			{
				body->completeUpdate(transform);
			}
		}
	}
	m_transformIsDirty = false;
}

void hkDefaultCompoundMeshBody::setIndexedTransforms(int startIndex, const hkMatrix4* matrices, int numMatrices)
{
    if (m_transformSet)
    {
        m_transformSet->setMatrices(startIndex, matrices, numMatrices);
        m_transformSetUpdated = true;
    }
}


void hkDefaultCompoundMeshBody::addToSystem(hkMeshSystem* meshSystem)
{
    completeUpdate();
	// Set position
	const int numBodies = m_bodies.getSize();
	for (int i = 0; i < numBodies; i++)
	{
        hkMeshBody* body = m_bodies[i];
        if (body)
        {
            meshSystem->addBody(body);
        }
	}
}

void hkDefaultCompoundMeshBody::removeFromSystem(hkMeshSystem* meshSystem)
{
	const int numBodies = m_bodies.getSize();
	for (int i = 0; i < numBodies; i++)
	{
        hkMeshBody* body = m_bodies[i];
        if (body)
        {
            meshSystem->removeBody(body);
        }
	}
}

void hkDefaultCompoundMeshBody::setIndexedInverseTransforms( int startIndex, const hkMatrix4* matrices, int numMatrices )
{
	if ( !m_transformSet )
	{
		return;
	}

	m_transformSet->setInverseMatrices(startIndex, matrices, numMatrices);
	m_transformSetUpdated = true;
}

void hkDefaultCompoundMeshBody::ensureInverseTransforms()
{
	if ( !m_transformSet )
	{
		return;
	}

	const int numMatrices = m_transformSet->m_matrices.getSize();

	if ( m_transformSet->m_inverseMatrices.getSize() == numMatrices )
	{
		return;
	}

	m_transformSet->m_inverseMatrices.setSize( numMatrices, hkMatrix4::getIdentity() );
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
