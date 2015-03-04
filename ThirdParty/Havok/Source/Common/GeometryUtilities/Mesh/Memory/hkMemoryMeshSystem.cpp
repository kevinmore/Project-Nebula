/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshSystem.h>
#include <Common/Base/Reflection/hkClass.h>

#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/GeometryUtilities/Mesh/Default/hkDefaultCompoundMeshShape.h>
#include <Common/GeometryUtilities/Mesh/Default/hkDefaultMeshMaterialRegistry.h>
#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshBody.h>
#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshMaterial.h>
#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshShape.h>
#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshTexture.h>
#include <Common/GeometryUtilities/Mesh/MultipleVertexBuffer/hkMultipleVertexBuffer.h>
#include <Common/GeometryUtilities/Mesh/Skin/hkSkinnedRefMeshShape.h>

HK_REFLECTION_DEFINE_STUB_VIRTUAL(hkMemoryMeshSystem, hkMeshSystem);

hkMemoryMeshSystem::hkMemoryMeshSystem()
{
	m_materialRegistry = new hkDefaultMeshMaterialRegistry;	
	m_materialRegistry->removeReference();
}

hkMemoryMeshSystem::~hkMemoryMeshSystem()
{
    freeResources();
	m_materialRegistry = HK_NULL;
}

hkMeshMaterialRegistry* hkMemoryMeshSystem::getMaterialRegistry() const
{
    return m_materialRegistry;
}

hkMeshShape* hkMemoryMeshSystem::createShape(const hkMeshSectionCinfo* sections, int numSections)
{
	if (hkDefaultCompoundMeshShape::hasIndexTransforms(sections, numSections))
	{
		// It has transform indices, we'll use a standard compound mesh to handle this case
		return hkDefaultCompoundMeshShape::createTransformIndexedShape(this, sections, numSections);
	}

    return new hkMemoryMeshShape(sections, numSections);
}

hkMeshShape* hkMemoryMeshSystem::createCompoundShape(const hkMeshShape*const* shapes, const hkMatrix4* transforms, int numShapes)
{
    return new hkDefaultCompoundMeshShape(shapes, transforms, numShapes);
}

//
//	hkMeshSystem implementation

hkMeshShape* hkMemoryMeshSystem::createCompoundShape(const hkMeshShape*const* shapes, const hkQTransform* transforms, int numShapes)
{
	// Call base class
	return hkMeshSystem::createCompoundShape(shapes, transforms, numShapes);
}


hkMeshBody* hkMemoryMeshSystem::createBody(const hkMeshShape* shapeIn, const hkMatrix4& mat, hkIndexedTransformSetCinfo* transformSet)
{
	if (shapeIn == HK_NULL)
	{
		return new hkMemoryMeshBody(this, shapeIn, mat, transformSet);
	}

	const hkClass* shapeType = shapeIn->getClassType();
	if (hkDefaultCompoundMeshShapeClass.equals(shapeType))
	{
		const hkDefaultCompoundMeshShape* shape = static_cast<const hkDefaultCompoundMeshShape*>(shapeIn);
		return new hkDefaultCompoundMeshBody(this, shape, mat, transformSet);
	}
	else if ( hkSkinnedRefMeshShapeClass.equals(shapeType) )
	{
		// In the case of skinrefs, create a body for the mesh buffer
		const hkSkinnedRefMeshShape* shape = static_cast<const hkSkinnedRefMeshShape*>(shapeIn);

		
		hkSkinnedMeshShape::BoneSection boneSection;
		shape->getSkinnedMeshShape()->getBoneSection( 0, boneSection );
		return new hkMemoryMeshBody(this, boneSection.m_meshBuffer, mat, transformSet);
	}

    return new hkMemoryMeshBody(this, shapeIn, mat, transformSet);
}

hkMeshVertexBuffer* hkMemoryMeshSystem::createVertexBuffer(const hkVertexFormat& vertexFormat, int numVertices)
{
	if (isSkinnedFormat(vertexFormat))
	{
		return createSkinnedVertexBuffer(vertexFormat, numVertices);
	}

	hkVertexFormat::SharingType type = vertexFormat.calculateSharingType();

	if (type == hkVertexFormat::SHARING_MIXTURE)
	{
		hkVertexFormat sharedFormat;
		hkVertexFormat instancedFormat;

		hkMultipleVertexBuffer* multipleBuffer = new hkMultipleVertexBuffer(vertexFormat, numVertices);

		const int numElements = vertexFormat.m_numElements;
		for (int i = 0; i < numElements; i++)
		{
			const hkVertexFormat::Element& ele = vertexFormat.m_elements[i];

			if (ele.m_flags.anyIsSet(hkVertexFormat::FLAG_NOT_SHARED))
			{
				multipleBuffer->addElement(0, instancedFormat.m_numElements);
				instancedFormat.addElement(ele);

			}
			else
			{
				multipleBuffer->addElement(1, sharedFormat.m_numElements);
				sharedFormat.addElement(ele);
			}
		}

		hkMemoryMeshVertexBuffer* sharedBuffer = new hkMemoryMeshVertexBuffer(sharedFormat, numVertices);
		hkMemoryMeshVertexBuffer* instancedBuffer = new hkMemoryMeshVertexBuffer(instancedFormat, numVertices);

		multipleBuffer->addVertexBuffer(instancedBuffer);
		multipleBuffer->addVertexBuffer(sharedBuffer);
		instancedBuffer->removeReference();
		sharedBuffer->removeReference();

		// Create it
		multipleBuffer->completeConstruction();

		return multipleBuffer;
	}
	else
	{
		return new hkMemoryMeshVertexBuffer(vertexFormat, numVertices);
	}
}

hkMeshVertexBuffer* hkMemoryMeshSystem::createVertexBuffer( const hkMeshVertexBuffer* templateVertexBuffer, int numVertices )
{
	return hkMeshSystem::createVertexBuffer( templateVertexBuffer, numVertices );
}

hkMeshVertexBuffer* hkMemoryMeshSystem::createSkinnedVertexBuffer(const hkVertexFormat& vertexFormat, int numVertices)
{
	// Skinned format
	hkVertexFormat skinnedFormat;
	hkVertexFormat otherFormat;

	hkMultipleVertexBuffer* buffer = new hkMultipleVertexBuffer(vertexFormat, numVertices);

	const int numElements = vertexFormat.m_numElements;
	for (int i = 0; i < numElements; i++)
	{
		const hkVertexFormat::Element& ele = vertexFormat.m_elements[i];
		hkVertexFormat::ComponentUsage usage = ele.m_usage;
		if ( (ele.m_subUsage != 0) && (usage != hkVertexFormat::USAGE_TEX_COORD) && (usage != hkVertexFormat::USAGE_COLOR) ) 
		{
			HK_WARN(0x12b35f71, "skinned vertex buffer does not support multiple components of type "<<usage);
			continue;
		}

		switch (usage)
		{
		case hkVertexFormat::USAGE_BLEND_MATRIX_INDEX:
			{
				buffer->addElement(1, skinnedFormat.m_numElements);

				// Decide element format based on the input format
				int elementIndex = vertexFormat.findElementIndex( hkVertexFormat::USAGE_BLEND_MATRIX_INDEX, 0 );
				const hkVertexFormat::Element& element = vertexFormat.m_elements[ elementIndex ];

				hkVertexFormat::ComponentType type;
				switch ( element.m_dataType )
				{
				case hkVertexFormat::TYPE_INT16:
				case hkVertexFormat::TYPE_UINT16:
					type = hkVertexFormat::TYPE_UINT16;
					break;
				default:
					type = hkVertexFormat::TYPE_UINT8;
					break;
				}

				skinnedFormat.addElement(hkVertexFormat::USAGE_BLEND_MATRIX_INDEX, type, ele.m_numValues, ele.m_flags.get());
				break;
			}
		case hkVertexFormat::USAGE_BLEND_WEIGHTS:
			{
				buffer->addElement(1, skinnedFormat.m_numElements);
				skinnedFormat.addElement(hkVertexFormat::USAGE_BLEND_WEIGHTS, hkVertexFormat::TYPE_UINT8, ele.m_numValues, ele.m_flags.get());
				break;
			}
		case hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED:
			{
				buffer->addElement(1, skinnedFormat.m_numElements);
				skinnedFormat.addElement(hkVertexFormat::USAGE_BLEND_WEIGHTS, hkVertexFormat::TYPE_UINT8, ele.m_numValues, ele.m_flags.get());
				break;
			}
		case hkVertexFormat::USAGE_POSITION:
		case hkVertexFormat::USAGE_NORMAL:
		case hkVertexFormat::USAGE_TANGENT:
		case hkVertexFormat::USAGE_BINORMAL:
		case hkVertexFormat::USAGE_TEX_COORD:
			{
				buffer->addElement(1, skinnedFormat.m_numElements);
				skinnedFormat.addElement(usage, hkVertexFormat::TYPE_VECTOR4, 1, ele.m_flags.get());
				// This is instanced for each usage
				otherFormat.addElement(usage, ele.m_dataType, ele.m_numValues, ele.m_flags.get() | hkVertexFormat::FLAG_NOT_SHARED);
				break;
			}
		case hkVertexFormat::USAGE_COLOR:
			{
				buffer->addElement(1, skinnedFormat.m_numElements);
				skinnedFormat.addElement(usage, hkVertexFormat::TYPE_ARGB32, 1, ele.m_flags.get());
				// This is instanced for each usage
				otherFormat.addElement(usage, ele.m_dataType, ele.m_numValues, ele.m_flags.get() | hkVertexFormat::FLAG_NOT_SHARED);
				break;
			}
		default:
			{
				buffer->addElement(0, otherFormat.m_numElements);
				otherFormat.addElement(ele);
				break;
			}
		}
	}

	// Create the vertex buffer
	hkMemoryMeshVertexBuffer* otherBuffer = new hkMemoryMeshVertexBuffer(otherFormat, numVertices);

	// Check the type is the same
	{
		hkVertexFormat otherFormatCreated;
		otherBuffer->getVertexFormat(otherFormatCreated);
		if (otherFormatCreated != otherFormat)
		{
			HK_WARN_ALWAYS(0x534534, "Couldn't create the vertex format for skinned mesh");
			otherBuffer->removeReference();
			buffer->removeReference();
			return HK_NULL;
		}
	}

	// Create the skinning vertex buffer
	hkMemoryMeshVertexBuffer* skinningBuffer = new hkMemoryMeshVertexBuffer(skinnedFormat, numVertices);

	buffer->addVertexBuffer(otherBuffer);
	otherBuffer->removeReference();
	buffer->addVertexBuffer(skinningBuffer);
	skinningBuffer->removeReference();

	buffer->completeConstruction();

	return buffer;
}

void hkMemoryMeshSystem::setMaterialRegistry(hkMeshMaterialRegistry* materialRegistry)
{
	m_materialRegistry = materialRegistry;	
}

bool hkMemoryMeshSystem::isSkinnedFormat(const hkVertexFormat& vertexFormat)
{
	const int numElements = vertexFormat.m_numElements;
	for (int i = 0; i < numElements; i++)
	{
		const hkVertexFormat::Element& ele = vertexFormat.m_elements[i];
		hkVertexFormat::ComponentUsage usage = ele.m_usage;

		if (ele.m_subUsage != 0) continue;

		switch (usage)
		{
		case hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED:
		case hkVertexFormat::USAGE_BLEND_WEIGHTS:
		case hkVertexFormat::USAGE_BLEND_MATRIX_INDEX:
			{
				return true;
			}
		default: break;
		}
	}

	return false;
}


void hkMemoryMeshSystem::findSuitableVertexFormat(const hkVertexFormat& format, hkVertexFormat& formatOut)
{
    HK_ASSERT(0xd8279a0d, format.isCanonicalOrder());
    formatOut = format;
}

hkMeshMaterial* hkMemoryMeshSystem::createMaterial()
{
    return new hkMemoryMeshMaterial;
}

//
//	Clones the given material.

hkMeshMaterial* hkMemoryMeshSystem::cloneMaterial(const hkMeshMaterial* srcMtl)
{
	if ( !hkMemoryMeshMaterialClass.equals(srcMtl->getClassType()) )
	{
		HK_WARN_ALWAYS(0xabba6d8e, "Invalid material type" );
		return HK_NULL;
	}

	const hkMemoryMeshMaterial* srcMeshMtl	= static_cast<const hkMemoryMeshMaterial*>(srcMtl);
	hkMemoryMeshMaterial* newMtl			= new hkMemoryMeshMaterial;

	*newMtl = *srcMeshMtl;
	return newMtl;
}

hkMeshTexture* hkMemoryMeshSystem::createTexture()
{
	return new hkMemoryMeshTexture();
}

void hkMemoryMeshSystem::freeResources()
{
	while (m_bodies.getSize() > 0)
	{
		hkPointerMap<hkMeshBody*, int>::Iterator iter = m_bodies.getIterator();
		HK_ASSERT(0x82740abb, m_bodies.isValid(iter));

		hkMeshBody* body = m_bodies.getKey(iter);
		removeBody(body);
	}

	if ( m_materialRegistry != HK_NULL )
	{
		m_materialRegistry->freeMaterials();
	}
}

void hkMemoryMeshSystem::addBody(hkMeshBody* bodyIn)
{
    HK_ASSERT(0x324232, bodyIn);

	// Check if it's already registered
	hkPointerMap<hkMeshBody*, int>::Iterator it = m_bodies.findKey( bodyIn );
	if ( m_bodies.isValid( it ) )
	{
		HK_WARN(0xd8279a37, "Mesh body already registered\n");
		return;
	}

    m_bodies.insert(bodyIn, 1);
    bodyIn->addReference();

	const hkClass* bodyType = bodyIn->getClassType();
	if (hkDefaultCompoundMeshBodyClass.equals(bodyType))
	{
		hkDefaultCompoundMeshBody* body = static_cast<hkDefaultCompoundMeshBody*>(bodyIn);
		body->addToSystem(this);
	}
}

void hkMemoryMeshSystem::removeBody(hkMeshBody* bodyIn)
{
	// Sanity check
	hkPointerMap<hkMeshBody*, int>::Iterator it = m_bodies.findKey( bodyIn );
	if ( !m_bodies.isValid( it ) )
	{
        HK_WARN(0xd8279a36, "Mesh body not registered\n");
		return;
	}

	// Get number of references
	int numReferences = m_bodies.getValue( it );

	// Remove only if it's the last reference
	// Bones can hold multiple references to the main skin
	if ( numReferences == 1 )
	{
		const hkClass* bodyType = bodyIn->getClassType();
		if (hkDefaultCompoundMeshBodyClass.equals(bodyType))
		{
			hkDefaultCompoundMeshBody* body = static_cast<hkDefaultCompoundMeshBody*>(bodyIn);
			body->removeFromSystem(this);
		}

		m_bodies.remove(bodyIn);
		bodyIn->removeReference();
	}
	else
	{
		// Remove one reference
		m_bodies.setValue( it, numReferences - 1 );
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
