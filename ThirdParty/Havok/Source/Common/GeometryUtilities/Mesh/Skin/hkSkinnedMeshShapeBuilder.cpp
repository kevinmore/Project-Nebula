/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/BitField/hkBitField.h>
#include <Common/Base/Memory/System/hkMemorySystem.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexBufferUtil/hkMeshVertexBufferUtil.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshShapeUtil/hkMeshShapeUtil.h>
#include <Common/GeometryUtilities/Mesh/Skin/hkSkinnedMeshShapeBuilder.h>

//
//	Constructor

hkSkinnedMeshBuilder::MeshSection::MeshSection()
:	m_meshIndex(-1)
,	m_surfaceIndex(-1)
,	m_subMeshIndex(-1)
,	m_boneSetIndex(-1)
,	m_hasDominants(false)
,	m_meshBoneSetId(BoneSetId::invalid())
{}

//
//	Copy constructor

hkSkinnedMeshBuilder::MeshSection::MeshSection(const MeshSection& other)
:	m_meshIndex(other.m_meshIndex)
,	m_surfaceIndex(other.m_surfaceIndex)
,	m_subMeshIndex(other.m_subMeshIndex)
,	m_boneSetIndex(other.m_boneSetIndex)
,	m_hasDominants(other.m_hasDominants)
,	m_meshBoneSetId(other.m_meshBoneSetId)
,	m_originalUsedVertices(other.m_originalUsedVertices)
{}

//
//	Compares two sections by surface index

HK_FORCE_INLINE bool HK_CALL hkSkinnedMeshBuilder::MeshSection::less(const MeshSection& sA, const MeshSection& sB)
{
	return (sA.m_surfaceIndex < sB.m_surfaceIndex);
}

//
//	Constructor

hkSkinnedMeshBuilder::hkSkinnedMeshBuilder(hkSkinnedMeshShape* skinnedMeshShape, hkMeshSystem* meshSystem, int maxNumBonesPerMesh)
:	m_maxNumBonesPerMesh(maxNumBonesPerMesh)
,	m_numTotalBones(0)
,	m_skinnedMeshShape(skinnedMeshShape)
,	m_meshSystem(meshSystem)
{}

//
//	Destructor

hkSkinnedMeshBuilder::~hkSkinnedMeshBuilder()
{}

//
//	Adds a mesh

void hkSkinnedMeshBuilder::addMesh(const hkMeshShape* origMesh, const hkQTransform& meshTransform, int numBones)
{
	// Make sure there are no more than maxBones per mesh section
	const int srcMeshIdx = m_sourceMeshes.getSize();
	hkArray<hkMeshBoneIndexMapping> boneMappings;
	const hkMeshShape* limitedMesh = (numBones > m_maxNumBonesPerMesh) ? hkMeshShapeUtil::createMeshWithLimitedNumBones(m_meshSystem, origMesh, m_maxNumBonesPerMesh, numBones, boneMappings) : origMesh;
	{
		m_sourceMeshes.pushBack(limitedMesh);
		m_sourceMeshTransforms.pushBack(meshTransform);

		if ( limitedMesh != origMesh )
		{
			// This is owned by us now
			limitedMesh->removeReference();
		}
	}

	// Add all bone transforms
	const int baseBoneId = m_numTotalBones;
	m_numTotalBones += numBones;

	// Add bone sets if any
	const int baseBoneSetId		= m_boneSets.getSize();
	const int numNewBoneSets	= boneMappings.getSize();
	for (int mi = 0; mi < numNewBoneSets; mi++)
	{
		hkMeshBoneIndexMapping& newSet	= m_boneSets.expandOne();
		const int numMappedBones		= boneMappings[mi].m_mapping.getSize();
		newSet.m_mapping.setSize(numMappedBones);

		for (int k = numMappedBones - 1; k >= 0; k--)
		{
			newSet.m_mapping[k] = (hkInt16)(baseBoneId + boneMappings[mi].m_mapping[k]);
		}
	}

	// If we have no bone mappings, add a default mapping with all bones
	if ( !numNewBoneSets )
	{
		hkMeshBoneIndexMapping& defaultMapping = m_boneSets.expandOne();
		defaultMapping.m_mapping.setSize(numBones);
		for (int bi = 0; bi < numBones; bi++)
		{
			defaultMapping.m_mapping[bi] = (hkInt16)(baseBoneId + bi);
		}
	}

	// Collect all mesh sections first. We'll assign them to different skin descriptors afterwards,
	// based on how well they fit the descriptors' bones
	const int numSections		= limitedMesh->getNumSections();
	MeshSection* newSections	= m_sourceMeshSections.expandBy(numSections);
	for (int si = 0; si < numSections; si++)
	{
		hkMeshSection srcSection;
		limitedMesh->lockSection(si, hkMeshShape::ACCESS_VERTEX_BUFFER | hkMeshShape::ACCESS_INDICES, srcSection);

		// Allocate a mesh section
		MeshSection& section	= newSections[si];
		section.m_meshIndex		= srcMeshIdx;
		section.m_subMeshIndex	= si;
		section.m_surfaceIndex	= addSurface(srcSection.m_material);
		section.m_hasDominants	= hkDisplacementMappingUtil::getDominantsMap(srcSection.m_material) != HK_NULL;
		section.m_boneSetIndex	= baseBoneSetId + (numNewBoneSets ? si : 0);
		calcRenderVertexRange(srcSection, section.m_originalUsedVertices);

		limitedMesh->unlockSection(srcSection);
	}
}

//
//	Computes the descriptors from the source meshes

void hkSkinnedMeshBuilder::computeDescriptors()
{
	hkBitField sectionBones(m_numTotalBones, hkBitFieldValue::ZERO);
	hkBitField descriptorBones(m_numTotalBones, hkBitFieldValue::ZERO);

	// Assign each one of the source mesh sections to a skin descriptor
	for (int si = m_sourceMeshSections.getSize() - 1; si >= 0; si--)
	{
		const MeshSection& srcSection			= m_sourceMeshSections[si];
		const hkMeshBoneIndexMapping& boneMap	= m_boneSets[srcSection.m_boneSetIndex];

		// Enable all section bones in the bit-field
		sectionBones.assignAll(0);
		for (int k = boneMap.m_mapping.getSize() - 1; k >= 0; k--)
		{
			sectionBones.set(boneMap.m_mapping[k]);
		}

		// Add the section to the "closest" descriptor (i.e. with maximum number of shared bones)
		int bestMergeScore		= 0x7FFFFFFF;
		int bestDescriptorIdx	= -1;
		for (int k = m_skinDescriptors.getSize() - 1; k >= 0; k--)
		{
			const SkinDescriptor& sd = m_skinDescriptors[k];

			descriptorBones.setOr(sd.m_usedBones, sectionBones);
			const int initialNumBones	= sd.m_usedBones.bitCount();
			const int finalNumBones		= descriptorBones.bitCount();
			const int deltaNumBones		= finalNumBones - initialNumBones;
			const int score				= (finalNumBones > m_maxNumBonesPerMesh) ? 0x7FFFFFFF : deltaNumBones;

			if ( score < bestMergeScore )
			{
				bestMergeScore		= score;
				bestDescriptorIdx	= k;
			}
		}

		// See if we must add a new descriptor
		if ( bestDescriptorIdx < 0 )
		{
			bestDescriptorIdx	= m_skinDescriptors.getSize();
			SkinDescriptor& sd	= m_skinDescriptors.expandOne();
			sd.m_hasDominants	= srcSection.m_hasDominants;
			sd.m_usedBones.setSizeAndFill(0, m_numTotalBones, 0);
		}

		// Add the mesh section to the descriptor
		{
			SkinDescriptor& sd	= m_skinDescriptors[bestDescriptorIdx];
			sd.m_hasDominants	= sd.m_hasDominants || srcSection.m_hasDominants;
			sd.m_sections.pushBack(srcSection);
			sd.m_usedBones.orWith(sectionBones);
		}
	}

	// Currently, mesh sections use the bone sets to map from section bone indices to global bone indices
	// We need to compute a mapping for each skin descriptor, that converts from "descriptor" indices to global bone indices
	for (int di = m_skinDescriptors.getSize() - 1; di >= 0; di--)
	{
		SkinDescriptor& sd = m_skinDescriptors[di];

		sd.m_localFromWorldBoneMap.m_mapping.setSize(m_numTotalBones, -1);
		for (int bi = 0; bi < m_numTotalBones; bi++)
		{
			if ( sd.m_usedBones.get(bi) )
			{
				// The global bone bi is used, add it to the descriptor
				sd.m_localFromWorldBoneMap.m_mapping[bi] = (hkInt16)sd.m_worldFromLocalBoneMap.m_mapping.getSize();
				sd.m_worldFromLocalBoneMap.m_mapping.pushBack((hkInt16)bi);
			}
		}
	}

	m_sourceMeshSections.clearAndDeallocate();
}

//
//	Adds a surface. Returns its index

int hkSkinnedMeshBuilder::addSurface(hkMeshMaterial* surface)
{
	const char* newName = surface->getName();
	for (int k = m_surfaces.getSize() - 1; k >= 0; k--)
	{
		hkMeshMaterial* uniqueSurface = m_surfaces[k];

		// Do not compare names
		const hkStringPtr origName = uniqueSurface->getName();
		uniqueSurface->setName(newName);
		const bool mtlsMatch = uniqueSurface->equals(surface);
		uniqueSurface->setName(origName);

		if ( mtlsMatch )
		{
			return k;
		}
	}

	// Not found, must add!
	m_surfaces.pushBack(surface);
	return m_surfaces.getSize() - 1;
}

//
//	Constructor

hkSkinnedMeshBuilder::SkinDescriptor::SkinDescriptor()
:	m_hasDominants(false)
{}

//
//	Counts the number of sub-meshes

int hkSkinnedMeshBuilder::SkinDescriptor::countSubmeshes() const
{
	int prevSurfaceId = -1;
	int numSubmeshes = 0;
	for (int i = 0; i < m_sections.getSize(); i++)
	{
		int currentSurfaceId = m_sections[i].m_surfaceIndex;
		if ( currentSurfaceId != prevSurfaceId )
		{
			prevSurfaceId = currentSurfaceId;
			numSubmeshes++;
		}
	}

	return numSubmeshes;
}
//
//	Computes the render vertex range

void hkSkinnedMeshBuilder::calcRenderVertexRange(const hkMeshSection& meshSection, hkBitField& usedVertsOut)
{
	// Alloc the bit-field
	const int numVerts = meshSection.m_vertexBuffer->getNumVertices();
	usedVertsOut.setSizeAndFill(0, numVerts, 0);

	// Iterate through all primitives and mark all used vertices
	hkMeshPrimitiveUtil::PrimitiveProvider pReader(meshSection.m_indices, meshSection.m_primitiveType, meshSection.m_indexType);
	for (int pi = 0; pi < meshSection.m_numPrimitives; pi++)
	{
		pReader.readPrimitive();
		usedVertsOut.set(pReader.m_a);
		usedVertsOut.set(pReader.m_b);
		usedVertsOut.set(pReader.m_c);
	}
}

//
//	Computes the shared vertex buffer format

void hkSkinnedMeshBuilder::computeVertexFormat(hkUint8 numBonesPerVertex)
{
	const int numMeshes = m_sourceMeshes.getSize();
	m_vfmt.m_numElements = 0;

	for (int meshIdx = 0; meshIdx < numMeshes; meshIdx++)
	{
		const hkMeshShape* meshShape = m_sourceMeshes[meshIdx];
		const int numSubMeshes = meshShape->getNumSections();

		for (int subMeshIdx = 0; subMeshIdx < numSubMeshes; subMeshIdx++)
		{
			hkMeshSection subMesh;
			meshShape->lockSection(subMeshIdx, hkMeshShape::ACCESS_VERTEX_BUFFER, subMesh);

			hkVertexFormat fmt;
			subMesh.m_vertexBuffer->getVertexFormat(fmt);
			hkMeshVertexBufferUtil::mergeVertexFormat(m_vfmt, fmt);

			meshShape->unlockSection(subMesh);
		}
	}

	// Add bone weights
	int boneWeightsIdx = m_vfmt.findElementIndex(hkVertexFormat::USAGE_BLEND_WEIGHTS, 0);
	if ( boneWeightsIdx < 0 )
	{
		boneWeightsIdx = m_vfmt.findElementIndex(hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED, 0);
		if ( boneWeightsIdx < 0 )
		{
			boneWeightsIdx = m_vfmt.m_numElements++;
			m_vfmt.m_elements[boneWeightsIdx].m_numValues = 0;
		}
	}
	{
		hkVertexFormat::Element& elt = m_vfmt.m_elements[boneWeightsIdx];
		elt.m_usage = hkVertexFormat::USAGE_BLEND_WEIGHTS;
		elt.m_subUsage	= 0;
		elt.m_dataType	= hkVertexFormat::TYPE_FLOAT32;
		elt.m_numValues	= hkMath::max2(numBonesPerVertex, elt.m_numValues);
	}

	// Add bone indices
	int boneIndicesIdx = m_vfmt.findElementIndex(hkVertexFormat::USAGE_BLEND_MATRIX_INDEX, 0);
	if ( boneIndicesIdx < 0 )
	{
		boneIndicesIdx = m_vfmt.m_numElements++;
		m_vfmt.m_elements[boneIndicesIdx].m_numValues = 0;
	}
	{
		hkVertexFormat::Element& elt = m_vfmt.m_elements[boneIndicesIdx];
		elt.m_usage = hkVertexFormat::USAGE_BLEND_MATRIX_INDEX;
		elt.m_subUsage	= 0;
		elt.m_dataType	= ( numMeshes > 255 ) ? hkVertexFormat::TYPE_INT16 : hkVertexFormat::TYPE_UINT8;
		elt.m_numValues	= hkMath::max2(numBonesPerVertex, elt.m_numValues);
	}

	m_vfmt.makeCanonicalOrder();
}

//
//	Computes the number of vertices in the given skin

int hkSkinnedMeshBuilder::computeNumVertices(SkinDescriptor& sd)
{
	const int numSections = sd.m_sections.getSize();
	int numVerts = 0;

	for (int si = 0; si < numSections; si++)
	{
		numVerts += sd.m_sections[si].m_originalUsedVertices.bitCount();
	}

	return numVerts;
}

//
//	Copies a sub-set of vertices from one locked vertex buffer to another

void hkSkinnedMeshBuilder::copyVertices(	const hkMeshVertexBuffer::LockedVertices& dstVerts, const hkVertexFormat& dstVtxFmt, int dstStartVertex, 
											const hkMeshVertexBuffer::LockedVertices& srcVerts, const hkVertexFormat& srcVtxFmt, const hkBitField& srcVertsToCopy)
{
	typedef hkMeshVertexBuffer::LockedVertices::Buffer Buffer;
	hkMeshVertexBuffer::LockedVertices src = srcVerts;
	hkMeshVertexBuffer::LockedVertices dst = dstVerts;

	// Find element indices
	hkLocalBuffer<int> bufIdx(src.m_numBuffers);
	for (int k = 0; k < src.m_numBuffers; k++)
	{
		Buffer& srcBuf	= src.m_buffers[k];
		bufIdx[k]		= dstVtxFmt.findElementIndex(srcBuf.m_element.m_usage, srcBuf.m_element.m_subUsage);
	}

	// Convert all source buffers.
	for (int vsi = 0, vdi = dstStartVertex; vsi < srcVertsToCopy.getSize(); vsi++)
	{
		if ( !srcVertsToCopy.get(vsi) )
		{
			continue;
		}

		// This vertex is in use, copy all its buffers
		for (int bsi = src.m_numBuffers - 1; bsi >= 0; bsi--)
		{
			const int bdi = bufIdx[bsi];
			if ( bdi < 0 )
			{
				continue;	// Element not present in the destination buffer
			}

			// Offset source and destination
			Buffer& srcBuf = src.m_buffers[bsi];
			Buffer& dstBuf = dst.m_buffers[bdi];
			srcBuf.m_start = hkAddByteOffset(srcVerts.m_buffers[bsi].m_start, vsi * srcBuf.m_stride);
			dstBuf.m_start = hkAddByteOffset(dstVerts.m_buffers[bdi].m_start, vdi * dstBuf.m_stride);
			HK_ASSERT(0x536e4b1a, srcBuf.m_element.m_usage == dstBuf.m_element.m_usage);
			HK_ASSERT(0x536e4b1a, srcBuf.m_element.m_subUsage == dstBuf.m_element.m_subUsage);

			hkMeshVertexBufferUtil::convert(srcBuf, dstBuf, 1);
		}

		vdi++;
	}
}

//
//	Applies the given transform to the given range of vertices

void hkSkinnedMeshBuilder::applyTransform(const hkQTransform& mtx, const hkMeshVertexBuffer::LockedVertices& verts, int startVertex, int numVertices)
{
	// If the transform is identity, do nothing
	if ( mtx.isApproximatelyEqual(hkQTransform::getIdentity(), hkReal(1.0e-6f)) )
	{
		return;
	}

	typedef hkMeshVertexBuffer::LockedVertices::Buffer Buffer;

	hkTransform tempTm;	tempTm.set(mtx.m_rotation, mtx.m_translation);
	hkMatrix4 tempMtx;	tempMtx.set(tempTm);

	// Offset verts
	hkMeshVertexBuffer::LockedVertices offsetVerts = verts;
	for (int k = offsetVerts.m_numBuffers - 1; k >= 0; k--)
	{
		Buffer& buf = offsetVerts.m_buffers[k];
		buf.m_start = hkAddByteOffset(buf.m_start, startVertex * buf.m_stride);

		hkMeshVertexBufferUtil::transform(buf, tempMtx, 0, numVertices);
	}
}

//
//	Computes the AABB of the given sub-set of vertices in the given vertex buffer

void hkSkinnedMeshBuilder::calcAabb(const hkMeshVertexBuffer::LockedVertices& lockedVerts, int startVertex, int numVertices, hkAabb& aabbOut)
{
	typedef hkMeshVertexBuffer::LockedVertices::Buffer Buffer;
	hkMeshVertexBuffer::LockedVertices offsetVerts = lockedVerts;

	for (int k = 0; k < offsetVerts.m_numBuffers ; k++)
	{
		Buffer& buf = offsetVerts.m_buffers[k];

		// Look for the positions buffer
		if ( buf.m_element.m_usage == hkVertexFormat::USAGE_POSITION )
		{
			// Offset positions buffer to startVertex
			buf.m_start = hkAddByteOffset(buf.m_start, startVertex * buf.m_stride);

			// Get positions
			hkArray<hkFloat32> verts;
			verts.setSize(numVertices << 2);
			hkMeshVertexBufferUtil::getElementVectorArray(buf, verts.begin(), numVertices);

			// Compute AABB
			{
				aabbOut.setEmpty();

				for (int i = 0; i < numVertices; i++)
				{
					hkVector4 v; v.load<4,HK_IO_NATIVE_ALIGNED>(&verts[i << 2]);
					aabbOut.includePoint(v);
				}

				aabbOut.m_min.zeroComponent<3>();
				aabbOut.m_max.zeroComponent<3>();
			}

			// Stop!
			break;
		}
	}
}

//
//	Utility functions

namespace hkSkinnedMeshBuilderImpl
{
	typedef hkDisplacementMappingUtil::DominantsBuffer	DominantsBuffer;
	typedef hkDisplacementMappingUtil::DominantInfo		DominantInfo;
	typedef hkMeshTexture::RawBufferDescriptor			Descriptor;

	/// Functor that copies the dominants from a texture into a buffer
	struct CopyDominants
	{
		HK_FORCE_INLINE CopyDominants(const hkMeshSection& meshSectionIn, DominantsBuffer& bufferOut, int dstVtxBase)
		:	m_srcDominants(HK_NULL)
		,	m_dstDominants(bufferOut)
		,	m_dstIdx(dstVtxBase)
		{
			// Get the material and its dominants texture
			const hkMeshMaterial* mtl	= meshSectionIn.m_material;
			hkMeshTexture* tex			= hkDisplacementMappingUtil::getDominantsMap(mtl);

			// Get raw data buffer from the texture
			int size = 0;
			hkMeshTexture::Format fmt;
			hkUint8* data;
			tex->getData(data, size, fmt);
			HK_ASSERT(0x30293db4, fmt == hkMeshTexture::RAW);

			// Get the descriptor
			Descriptor& d = *reinterpret_cast<Descriptor*>(data);
			m_srcDominants = data + d.m_offset;
		}

		HK_FORCE_INLINE void operator()(int srcVtxIdx)
		{
			const int stride	= sizeof(DominantInfo);
			const hkUint8* src	= &m_srcDominants[srcVtxIdx * stride];
			DominantInfo& dst	= m_dstDominants[m_dstIdx];

			dst.load(src);
			m_dstIdx++;
		}

		hkUint8* m_srcDominants;
		DominantsBuffer& m_dstDominants;
		int m_dstIdx;
	};
}

//
//	Fills the provided skinned vertex buffer with the data from the given skin descriptor

void hkSkinnedMeshBuilder::fillSkinnedVertexBuffer(VertexBuffer* skinnedBuffer, int vbOffset, int numVbVerts, const SkinDescriptor& sd)
{
	// Lock the target buffer
	hkMeshVertexBuffer::LockInput dstLockInput;
	dstLockInput.m_lockFlags	= hkMeshVertexBuffer::ACCESS_WRITE_DISCARD;
	dstLockInput.m_startVertex	= vbOffset;
	dstLockInput.m_numVertices	= numVbVerts;

	hkMeshVertexBuffer::LockedVertices dstLockedVerts;
	hkMeshVertexBuffer* skinnedVb = skinnedBuffer->m_vb;
	skinnedVb->lock(dstLockInput, dstLockedVerts);

	const int numMeshSections = sd.m_sections.getSize();
	int startVtx = 0;
	for (int si = 0; si < numMeshSections; si++)
	{
		// Get source vertex buffer
		const MeshSection& section = sd.m_sections[si];
		const hkMeshShape* dynMesh = m_sourceMeshes[section.m_meshIndex];

		// Get the sub-mesh
		hkMeshSection subMesh;
		dynMesh->lockSection(section.m_subMeshIndex, hkMeshShape::ACCESS_VERTEX_BUFFER, subMesh);
		{
			hkMeshVertexBuffer* srcVb = subMesh.m_vertexBuffer;

			// Get source format
			hkVertexFormat srcVFmt;
			srcVb->getVertexFormat(srcVFmt);

			// Lock it and copy
			hkMeshVertexBuffer::LockInput srcLockInput;
			srcLockInput.m_lockFlags	= hkMeshVertexBuffer::ACCESS_READ;
			const int numCopiedVerts	= section.m_originalUsedVertices.bitCount();
			hkMeshVertexBuffer::LockedVertices srcLockedVerts;

			srcVb->lock(srcLockInput, srcLockedVerts);
			{
				HK_ASSERT(0x4b26085d, startVtx + numCopiedVerts <= numVbVerts);
				copyVertices(dstLockedVerts, m_vfmt, startVtx, srcLockedVerts, srcVFmt, section.m_originalUsedVertices);
			}
			srcVb->unlock(srcLockedVerts);

			// Copy dominants
			if ( sd.m_hasDominants && section.m_hasDominants )
			{
				hkSkinnedMeshBuilderImpl::CopyDominants fun(subMesh, skinnedBuffer->m_dominants, vbOffset + startVtx);
				section.m_originalUsedVertices.forEachBitSet(fun);
			}

			// Transform if necessary
			applyTransform(m_sourceMeshTransforms[section.m_meshIndex], dstLockedVerts, startVtx, numCopiedVerts);
			startVtx += numCopiedVerts;
		}
		dynMesh->unlockSection(subMesh);
	}

	// Unlock the target buffer
	HK_ASSERT(0x2fc2548d, numVbVerts == startVtx);
	skinnedVb->unlock(dstLockedVerts);
}

//
//	Converts an array of indices from mesh-section bone-set "space" to skin-descriptor "space"

void hkSkinnedMeshBuilder::convertBoneIndices(hkArray<int>& boneSetIndices, hkArray<hkFloat32>& boneSetWeights, int boneSetId, const SkinDescriptor& sd) const
{
	const hkMeshBoneIndexMapping& worldFromBoneSet		= m_boneSets[boneSetId];
	const hkMeshBoneIndexMapping& descriptorFromWorld	= sd.m_localFromWorldBoneMap;

	HK_ASSERT(0x3700b29, boneSetIndices.getSize() == boneSetWeights.getSize());
	for (int k = boneSetIndices.getSize() - 1; k >= 0; k--)
	{
		hkFloat32& boneSetWeight	= boneSetWeights[k];
		hkInt16 descriptorLocalIdx	= 0;	// Use a valid bone if the weight is 0!

		if ( boneSetWeight > 0.0f )
		{
			const hkInt16 boneSetLocalIdx	= (hkInt16)boneSetIndices[k];

			// Check for a valid remapping
			if ( (boneSetLocalIdx >= 0) && (boneSetLocalIdx < worldFromBoneSet.m_mapping.getSize()) )
			{
				const hkInt16 globalIdx	= worldFromBoneSet.m_mapping[boneSetLocalIdx];
				descriptorLocalIdx		= descriptorFromWorld.m_mapping[globalIdx];
			}
			else
			{
				// Invalid remapping (can happen if the original weight was zero, saved as 3 quantized weights and unquantized 4th != 0), set weight to zero!
				boneSetWeight = 0.0f;
			}
		}

		boneSetIndices[k] = descriptorLocalIdx;
	}
}

//
//	Set-up the bone weights & indices

void hkSkinnedMeshBuilder::createBoneWeightsAndIndices(hkMeshVertexBuffer* skinnedVb, int vbOffset, int numVbVerts, const SkinDescriptor& sd)
{
	// Lock target buffer
	hkMeshVertexBuffer::LockInput dstLockInput;
	dstLockInput.m_startVertex	= vbOffset;
	dstLockInput.m_numVertices	= numVbVerts;

	hkMeshVertexBuffer::LockedVertices dstLockedVerts;
	skinnedVb->lock(dstLockInput, dstLockedVerts);

	// Find the indices
	const int dstIdxWeights	= hkMath::max2(m_vfmt.findElementIndex(hkVertexFormat::USAGE_BLEND_WEIGHTS, 0), m_vfmt.findElementIndex(hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED, 0));
	const int dstIdxMats	= m_vfmt.findElementIndex(hkVertexFormat::USAGE_BLEND_MATRIX_INDEX, 0);

	hkArray<int> boneIndices;
	hkArray<hkFloat32> boneWeights;
	hkArray<int> boneIndicesStrided;
	const int numMeshSections = sd.m_sections.getSize();
	int startVtx = 0;
	for (int si = 0; si < numMeshSections; si++)
	{
		// Get source vertex buffer
		const MeshSection& section	= sd.m_sections[si];
		const hkMeshShape* dynMesh	= m_sourceMeshes[section.m_meshIndex];

		// Get the sub-mesh
		hkMeshSection subMesh;
		dynMesh->lockSection(section.m_subMeshIndex, hkMeshShape::ACCESS_VERTEX_BUFFER, subMesh);
		{
			hkMeshVertexBuffer* srcVb = subMesh.m_vertexBuffer;

			// Get source format
			hkVertexFormat srcVFmt;
			srcVb->getVertexFormat(srcVFmt);

			// If we have source skinning data, use it. Otherwise build it now
			const int srcIdxWeights		= hkMath::max2(srcVFmt.findElementIndex(hkVertexFormat::USAGE_BLEND_WEIGHTS, 0), srcVFmt.findElementIndex(hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED, 0));
			const int srcIdxMats		= srcVFmt.findElementIndex(hkVertexFormat::USAGE_BLEND_MATRIX_INDEX, 0);
			const int numCopiedVerts	= section.m_originalUsedVertices.bitCount();

			boneIndices.setSize(numCopiedVerts << 2);
			boneWeights.setSize(numCopiedVerts << 2);
			// Set them all to default values
			hkString::memSet4(boneIndices.begin(), 0, boneIndices.getSize());
			hkString::memSet4(boneWeights.begin(), 0, boneWeights.getSize());

			if ( srcIdxMats >= 0 )
			{
				// Lock it and copy
				hkMeshVertexBuffer::LockInput srcLockInput;
				srcLockInput.m_lockFlags	= hkMeshVertexBuffer::ACCESS_READ;
				hkMeshVertexBuffer::LockedVertices srcLockedVerts;

				srcVb->lock(srcLockInput, srcLockedVerts);
				{
					HK_ASSERT(0x4b26085d, startVtx + numCopiedVerts <= numVbVerts);
					const int numIndicesValues = srcVFmt.m_elements[srcIdxMats].m_numValues;
					const int boneIndicesStridedSize = numCopiedVerts * numIndicesValues;
					boneIndicesStrided.setSize(boneIndicesStridedSize);
					hkMeshVertexBufferUtil::getElementIntArray(srcLockedVerts, srcIdxMats, section.m_originalUsedVertices, boneIndicesStrided.begin(), boneIndicesStridedSize);
					hkMeshVertexBufferUtil::getElementVectorArray(srcLockedVerts, srcIdxWeights, section.m_originalUsedVertices, boneWeights.begin());
					hkMeshVertexBufferUtil::stridedCopy(boneIndicesStrided.begin(), numIndicesValues*sizeof(int), boneIndices.begin(), 4*sizeof(int), numIndicesValues*sizeof(int), numCopiedVerts);
				}
				srcVb->unlock(srcLockedVerts);
			}
			else
			{
				// Init default weights
				for (int k = 0; k < boneWeights.getSize(); k += 4)
				{
					boneWeights[k] = 1;
				}
			}

			// Convert indices from mesh section bone set to skin descriptor bone set
			convertBoneIndices(boneIndices, boneWeights, section.m_boneSetIndex, sd);

			// Stride according to the destination vertex format
			const int numIndicesValues	= m_vfmt.m_elements[dstIdxMats].m_numValues;
			boneIndicesStrided.setSize(numCopiedVerts * numIndicesValues);
			hkMeshVertexBufferUtil::stridedCopy(boneIndices.begin(), 4*sizeof(int), boneIndicesStrided.begin(), numIndicesValues*sizeof(int), numIndicesValues*sizeof(int), numCopiedVerts);

			// Store in the destination buffer
			hkMeshVertexBufferUtil::setElementIntArray(dstLockedVerts, dstIdxMats, startVtx, boneIndicesStrided.begin(), numCopiedVerts);
			hkMeshVertexBufferUtil::setElementVectorArray(dstLockedVerts, dstIdxWeights, startVtx, boneWeights.begin(), numCopiedVerts);

			startVtx += numCopiedVerts;
		}
		dynMesh->unlockSection(subMesh);
	}

	HK_ASSERT(0x2fc2548d, numVbVerts == startVtx);
	skinnedVb->unlock(dstLockedVerts);
}

//
//	Compute common index buffer format

void hkSkinnedMeshBuilder::computeIndexFormat(const SkinDescriptor& sd, hkMergeMeshPrimitvesCalculator& mpc)
{
	const int numMeshSections = sd.m_sections.getSize();
	for (int si = 0; si < numMeshSections; si++)
	{
		const MeshSection& section = sd.m_sections[si];
		const hkMeshShape* dynMesh = m_sourceMeshes[section.m_meshIndex];

		hkMeshSection subMesh;
		dynMesh->lockSection(section.m_subMeshIndex, hkMeshShape::ACCESS_INDICES, subMesh);
		mpc.add(subMesh.m_numPrimitives, subMesh.m_primitiveType, subMesh.m_indexType);
		dynMesh->unlockSection(subMesh);
	}
}

//
//	Creates the mesh for the given descriptor. The indices are not properly set at this stage

hkMeshShape* hkSkinnedMeshBuilder::createMesh(const SkinDescriptor* HK_RESTRICT descriptors, int numDescriptors, const hkMergeMeshPrimitvesCalculator& mpc, VertexBuffer* bufferOut)
{
	// Alloc temporary buffers
	hkArray<int> tempIdxBuffer;
	hkArray<hkMeshMaterial*> tempMtls;
	{
		const int numTotalIndices = mpc.getNumIndices();
		tempIdxBuffer.setSize(numTotalIndices, 0);
	}

	// Prepare the sub-sections
	hkArray<hkMeshSectionCinfo> cinfos;
	int numSkinSubmeshes = 0;
	for (int di = 0; di < numDescriptors; di++)
	{
		numSkinSubmeshes += descriptors[di].countSubmeshes();
	}
	cinfos.setSize(numSkinSubmeshes);

	// Copy the indices
	int prevSurfaceIdx			= -1;
	int skinSectionStartVertex	= 0;
	int skinSectionStartIndex	= 0;
	int skinSectionIdx			= -1;
	int skinSectionNumVertices	= 0;
	int skinSectionNumIndices	= 0;

	hkMeshSectionCinfo* dstSubMesh	= HK_NULL;
	hkMeshVertexBuffer* vb			= bufferOut->m_vb;
	const int numVbVerts = vb->getNumVertices();
	for (int di = 0, si = 0, currentIndexBase = 0, currentVertexBase = 0; ; si++)
	{
		bool switchSection = false;
		if ( si >= descriptors[di].m_sections.getSize() )
		{
			// Current mesh section past the current descriptor size!
			si = 0;
			di++;
			switchSection = true;
			if ( di >= numDescriptors )
			{
				break;
			}
		}

		// See if we need to switch the skinned mesh section
		const SkinDescriptor& sd	= descriptors[di];
		const MeshSection& section	= sd.m_sections[si];
		const int currentSurfaceIdx = section.m_surfaceIndex;
		if ( switchSection || (prevSurfaceIdx != currentSurfaceIdx) )
		{
			// New skinned sub-mesh!
			// Finalize previous sub-mesh
			if ( dstSubMesh )
			{
				dstSubMesh->m_numPrimitives = hkMeshPrimitiveUtil::calculateNumPrimitives(dstSubMesh->m_primitiveType, skinSectionNumIndices);
			}

			// Initialize new sub-mesh
			prevSurfaceIdx = currentSurfaceIdx;
			skinSectionIdx++;
			skinSectionStartVertex		+= skinSectionNumVertices;
			skinSectionStartIndex		+= skinSectionNumIndices;
			skinSectionNumVertices		= 0;
			skinSectionNumIndices		= 0;

			dstSubMesh = &cinfos[skinSectionIdx];
			dstSubMesh->m_vertexBuffer		= vb;

			// Set-up displacements
			hkMeshMaterial* dstMtl;
			{
				dstMtl = m_surfaces[currentSurfaceIdx];

				// Check whether the material had displacement maps
				if ( hkDisplacementMappingUtil::getDisplacementMap(dstMtl) )
				{
					hkMeshTexture* dMap	= bufferOut->m_dominants.realize(m_meshSystem);

					// If the vb has dominants, we need to clone the original material and replace its dominants.
					if ( dMap )
					{
						dstMtl = hkDisplacementMappingUtil::duplicateMaterial(m_meshSystem, dstMtl);
						hkDisplacementMappingUtil::setDominantsMap(dstMtl, dMap);
						tempMtls.pushBack(dstMtl);
					}
				}
			}

			dstSubMesh->m_material			= dstMtl;
			dstSubMesh->m_primitiveType		= mpc.getPrimitiveType();
			dstSubMesh->m_numPrimitives		= 0;
			dstSubMesh->m_indexType			= mpc.getIndexType(numVbVerts);
			dstSubMesh->m_indices			= &tempIdxBuffer[skinSectionStartIndex];
			dstSubMesh->m_vertexStartIndex	= skinSectionStartVertex;
			dstSubMesh->m_transformIndex	= -1;
		}

		// Copy this mesh section into the skinned section
		const hkMeshShape* srcMesh = m_sourceMeshes[section.m_meshIndex];
		hkMeshSection srcSubMesh;
		srcMesh->lockSection(section.m_subMeshIndex, hkMeshShape::ACCESS_INDICES, srcSubMesh);

		// Increment the index buffer pointer
		const int deltaIndices	= hkMeshPrimitiveUtil::calculateNumIndices(mpc.getPrimitiveType(), srcSubMesh.m_numPrimitives);
		const int numOrigVerts	= section.m_originalUsedVertices.bitCount();
		currentVertexBase		+= numOrigVerts;
		currentIndexBase		+= deltaIndices;
		skinSectionNumVertices	+= numOrigVerts;
		skinSectionNumIndices	+= deltaIndices;

		srcMesh->unlockSection(srcSubMesh);
	}

	// Finalize previous sub-mesh
	if ( dstSubMesh )
	{
		dstSubMesh->m_numPrimitives = hkMeshPrimitiveUtil::calculateNumPrimitives(dstSubMesh->m_primitiveType, skinSectionNumIndices);
	}

	// Create the mesh shape, clean-up and return
	hkMeshShape* retMesh = m_meshSystem->createShape(cinfos.begin(), cinfos.getSize());
	hkReferencedObject::removeReferences(tempMtls.begin(), tempMtls.getSize());
	return retMesh;
}

//
//	Fills the provided skinned index buffer with the data from the given skin descriptor

void hkSkinnedMeshBuilder::fillSkinnedIndexBuffer(const SkinDescriptor* HK_RESTRICT descriptors, int numDescriptors, hkMeshShape* skinnedMesh)
{
	// Copy the indices
	int prevSurfaceIdx			= -1;
	int skinSectionStartVertex	= 0;
	int skinSectionStartIndex	= 0;
	int skinSectionIdx			= -1;
	int skinSectionNumVertices	= 0;
	int skinSectionNumIndices	= 0;

	hkMeshSection lockedSubMesh;
	hkMeshSection* dstSubMesh	= HK_NULL;
	hkUint8* dstIndices			= HK_NULL;

	hkArray<int> vtxRemap;
	int totalNumPrimitives = 0;
	for (int di = 0, si = 0, currentIndexBase = 0, currentVertexBase = 0; ; si++)
	{
		// See if we need to switch the skinned mesh section
		bool switchSection = false;
		if ( si >= descriptors[di].m_sections.getSize() )
		{
			si = 0;
			di++;
			switchSection = true;
			if ( di >= numDescriptors )
			{
				break;
			}
		}

		// See if we need to switch the skinned mesh section
		const SkinDescriptor& sd	= descriptors[di];
		const MeshSection& section	= sd.m_sections[si];
		const int currentSurfaceIdx = section.m_surfaceIndex;
		if ( switchSection || (prevSurfaceIdx != currentSurfaceIdx) )
		{
			// New skinned sub-mesh!
			// Finalize previous sub-mesh
			if ( dstSubMesh )
			{
				skinnedMesh->unlockSection(lockedSubMesh);
				dstSubMesh = HK_NULL;
				dstIndices = HK_NULL;
			}

			// Initialize new sub-mesh
			prevSurfaceIdx = currentSurfaceIdx;
			skinSectionIdx++;
			skinnedMesh->lockSection(skinSectionIdx, hkMeshShape::ACCESS_INDICES | hkMeshShape::ACCESS_VERTEX_BUFFER, lockedSubMesh);
			dstSubMesh = &lockedSubMesh;
			dstIndices = (hkUint8*)const_cast<void*>(dstSubMesh->m_indices);
			skinSectionStartVertex	+= skinSectionNumVertices;
			skinSectionStartIndex	+= skinSectionNumIndices;
			skinSectionNumVertices	= 0;
			skinSectionNumIndices	= 0;
		}

		// Copy this mesh section into the skinned section
		const hkMeshShape* srcMesh = m_sourceMeshes[section.m_meshIndex];
		hkMeshSection srcSubMesh;
		srcMesh->lockSection(section.m_subMeshIndex, hkMeshShape::ACCESS_INDICES, srcSubMesh);

		// Compute a vertex remap table
		const int numSectionVerts	= section.m_originalUsedVertices.bitCount();
		vtxRemap.setSize(section.m_originalUsedVertices.getSize());
		for (int svi = 0, dvi = currentVertexBase; svi < section.m_originalUsedVertices.getSize(); svi++)
		{
			vtxRemap[svi] = section.m_originalUsedVertices.get(svi) ? dvi++ : -1;
		}
		
		// Copy & convert the indices
		hkMeshPrimitiveUtil::copyAndRemapPrimitives(srcSubMesh.m_indices, srcSubMesh.m_primitiveType, srcSubMesh.m_indexType, srcSubMesh.m_numPrimitives,
													dstIndices, dstSubMesh->m_primitiveType, dstSubMesh->m_indexType, vtxRemap.begin());
		totalNumPrimitives += srcSubMesh.m_numPrimitives;

		// Compute Aabb
		hkAabb meshSectionAabb; meshSectionAabb.setEmpty();
		{
			hkMeshVertexBuffer* skinnedVb = lockedSubMesh.m_vertexBuffer;
			hkMeshVertexBuffer::LockInput skinLockInput;
			skinLockInput.m_startVertex	= currentVertexBase;
			skinLockInput.m_numVertices	= numSectionVerts;
			hkMeshVertexBuffer::LockedVertices skinLockedVerts;
			
			skinnedVb->lock(skinLockInput, skinLockedVerts);
			calcAabb(skinLockedVerts, 0, numSectionVerts, meshSectionAabb);
			skinnedVb->unlock(skinLockedVerts);
		}

		// Increment the index buffer pointer
		const int deltaIndices	= hkMeshPrimitiveUtil::calculateNumIndices(dstSubMesh->m_primitiveType, srcSubMesh.m_numPrimitives);

		// Save the part
		HK_ASSERT(0x74fa6d26, section.m_meshBoneSetId.isValid());
		addPart(section.m_meshBoneSetId, (hkUint16)skinSectionIdx, currentVertexBase, numSectionVerts, currentIndexBase - skinSectionStartIndex, deltaIndices, meshSectionAabb);
		currentVertexBase		+= numSectionVerts;
		currentIndexBase		+= deltaIndices;
		skinSectionNumVertices	+= numSectionVerts;
		skinSectionNumIndices	+= deltaIndices;
		dstIndices				+= hkMeshPrimitiveUtil::getIndexEntrySize(dstSubMesh->m_indexType) * deltaIndices;

		srcMesh->unlockSection(srcSubMesh);
	}

	// Finalize previous sub-mesh
	if ( dstSubMesh )
	{
		skinnedMesh->unlockSection(lockedSubMesh);
	}
}

//
//	Adds a new part

void hkSkinnedMeshBuilder::addPart(BoneSetId boneSetId, hkUint16 meshSectionIndex, int startVtx, int numVertices, int startIdx, int numIndices, const hkAabb& aabb)
{
	hkSkinnedMeshShape::Part part;
	part.m_startVertex		= startVtx;
	part.m_numVertices		= numVertices;
	part.m_startIndex		= startIdx;
	part.m_numIndices		= numIndices;
	part.m_boneSetId		= boneSetId;
	part.m_meshSectionIndex	= meshSectionIndex;

	hkVector4 vCenter;			aabb.getCenter(vCenter);
	hkVector4 vExtents;			aabb.getExtents(vExtents);
	const hkSimdReal diameter	= vExtents.length<3>();
	hkSimdReal radius;			radius.setMul(hkSimdReal_Inv2, diameter);	
	part.m_boundingSphere.setXYZ_W(vCenter, radius);

	m_skinnedMeshShape->addPart(part);
}

//
//	Adds a bone section for the given skin descriptor

void hkSkinnedMeshBuilder::addBoneSection(hkMeshShape* boneSectionMesh, SkinDescriptor& sd)
{
	// Add all bone sets first
	BoneSetId firstBoneSetId(0x7FFF);
	BoneSetId lastBoneSetId(0);
	for (int k = 0; k < sd.m_sections.getSize(); k++)
	{
		MeshSection& section = sd.m_sections[k];
		const hkMeshBoneIndexMapping& srcBoneSet = m_boneSets[section.m_boneSetIndex];

		section.m_meshBoneSetId = m_skinnedMeshShape->addBoneSet(srcBoneSet.m_mapping.begin(), srcBoneSet.m_mapping.getSize());
		firstBoneSetId			= hkMath::min2(firstBoneSetId, section.m_meshBoneSetId);
		lastBoneSetId			= hkMath::max2(lastBoneSetId, section.m_meshBoneSetId);
	}

	// Add bone section
	const hkInt16 numBoneSets = (hkInt16)(lastBoneSetId.value() - firstBoneSetId.value() + 1);
	m_skinnedMeshShape->addBoneSection(boneSectionMesh, firstBoneSetId, numBoneSets);
}

//
//	Constructor

hkSkinnedMeshBuilder::VertexBuffer::VertexBuffer(hkMeshSystem* meshSystem, const hkVertexFormat& vFmt, int numVerts, bool hasDominants)
:	hkReferencedObject()
{
	m_vb.setAndDontIncrementRefCount(meshSystem->createVertexBuffer(vFmt, numVerts));

	if ( hasDominants )
	{
		m_dominants.alloc(numVerts);
	}
}

//
//	Builds the skins

void hkSkinnedMeshBuilder::build(bool buildSingleMesh, hkUint8 numBonesPerVertex)
{
	computeDescriptors();
	computeVertexFormat(numBonesPerVertex);

	// Sort all skin descriptors' sections by material
	const int numSkins	= m_skinDescriptors.getSize();
	if ( !numSkins )
	{
		return;
	}

	int numTotalVerts	= 0;
	bool hasDominants	= false;
	hkLocalBuffer<int>	numSkinVerts(numSkins);
	for (int i = 0; i < numSkins; i++)
	{
		SkinDescriptor& sd = m_skinDescriptors[i];
		hkAlgorithm::quickSort(sd.m_sections.begin(), sd.m_sections.getSize(), MeshSection::less);

		// Compute the number of vertices in this skin
		numSkinVerts[i] = computeNumVertices(sd);
		numTotalVerts += numSkinVerts[i];
		hasDominants	= hasDominants || sd.m_hasDominants;
	}

	// Create vertex buffers
	hkRefPtr<VertexBuffer> globalVb = HK_NULL;
	if ( buildSingleMesh )
	{
		globalVb.setAndDontIncrementRefCount(new VertexBuffer(m_meshSystem, m_vfmt, numTotalVerts, hasDominants));
	}

	hkLocalArray< hkRefPtr<VertexBuffer> > vbs(numSkins);
	{
		// If we must create a single mesh, allocate a global vb
		vbs.setSize(numSkins, globalVb);
		
		for (int i = 0, vtxBase = 0; i < numSkins; vtxBase += numSkinVerts[i], i++)
		{
			const SkinDescriptor& sd = m_skinDescriptors[i];

			// Allocate the target vertex buffer. The vertex format might be changed, so grab it back!
			if ( !vbs[i] )
			{
				vbs[i].setAndDontIncrementRefCount(new VertexBuffer(m_meshSystem, m_vfmt, numSkinVerts[i], sd.m_hasDominants));
				vtxBase = 0;
			}

			vbs[i]->m_vb->getVertexFormat(m_vfmt);
			fillSkinnedVertexBuffer(vbs[i], vtxBase, numSkinVerts[i], sd);

			// Set-up the bone weights & indices
			createBoneWeightsAndIndices(vbs[i]->m_vb, vtxBase, numSkinVerts[i], sd);
		}
	}

	// Create mesh(es)
	if ( globalVb )
	{
		// Compute common index buffer format
		hkMergeMeshPrimitvesCalculator mpc;

		for (int i = 0; i < numSkins; i++)
		{
			const SkinDescriptor& sd = m_skinDescriptors[i];
			computeIndexFormat(sd, mpc);
		}

		// Create the global mesh
		hkMeshShape* skinnedMesh = createMesh(m_skinDescriptors.begin(), m_skinDescriptors.getSize(), mpc, globalVb);

		// Add all bone sections
		for (int i = 0; i < numSkins; i++)
		{
			addBoneSection(skinnedMesh, m_skinDescriptors[i]);
		}

		// Fill the index buffer
		fillSkinnedIndexBuffer(m_skinDescriptors.begin(), m_skinDescriptors.getSize(), skinnedMesh);
		m_skinnedMeshShape->sortParts();

		skinnedMesh->removeReference();
	}
	else
	{
		for (int i = 0, vtxBase = 0; i < numSkins; vtxBase += numSkinVerts[i], i++)
		{
			SkinDescriptor& sd = m_skinDescriptors[i];

			// Compute common index buffer format
			hkMergeMeshPrimitvesCalculator mpc;
			computeIndexFormat(sd, mpc);

			// Start a new bone section
			hkMeshShape* skinnedMesh = createMesh(&sd, 1, mpc, vbs[i]);
			addBoneSection(skinnedMesh, sd);
			skinnedMesh->removeReference();

			// Fill the index buffer
			fillSkinnedIndexBuffer(&sd, 1, skinnedMesh);
			m_skinnedMeshShape->sortParts();

			// Set-up the name
			const char* fullSkinName = m_skinnedMeshShape->getName();
			hkStringBuf boneSectionName;
			boneSectionName.printf("%s__boneSection__%d", fullSkinName ? fullSkinName : "(null)", i);
			skinnedMesh->setName(boneSectionName.cString());
		}
	}
}

void hkSkinnedMeshBuilder::getSkinningInfo( hkArray<SkinningInfo>& sectionsOut )
{
	// All the sections are parts share the same vertex buffer,
	// it's safe to just add them after each other.

	int actVertex = 0;
	const int numDescriptors = m_skinDescriptors.getSize();
	for ( int d=0; d<numDescriptors; d++ )
	{
		const hkSkinnedMeshBuilder::SkinDescriptor& descriptor = m_skinDescriptors[d];

		const int numSections = descriptor.m_sections.getSize();
		for ( int s=0; s<numSections; s++ )
		{
			const hkSkinnedMeshBuilder::MeshSection& section = descriptor.m_sections[s];

			// Advance position
			actVertex += section.m_originalUsedVertices.bitCount();

			// Store new section
			SkinningInfo& info = sectionsOut.expandOne();
			info.m_lastVertex  = actVertex-1;
			info.m_boneIndex   = m_boneSets[section.m_boneSetIndex].m_mapping[0];
		}
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
