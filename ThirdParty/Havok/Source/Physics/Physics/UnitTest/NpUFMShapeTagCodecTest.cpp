/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Serialize/Util/hkSerializeUtil.h>
#include <Physics/Physics/Collide/Shape/Composite/Compound/Static/hknpStaticCompoundShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Compressed/hknpCompressedMeshShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Sphere/hknpSphereShape.h>
#include <Physics/Physics/Collide/Shape/hknpShapeCollector.h>
#include <Physics/Physics/Collide/Shape/TagCodec/UFM/hknpUFMShapeTagCodec.h>


void testMaterialDescriptors()
{
	hknpMaterialLibrary materialLibrary;

	// Named descriptor
	{
		hknpMaterialDescriptor descriptor;
		descriptor.m_name = "NamedMaterial";

		// Not present
		{
			HK_TEST_ASSERT(0x623f8562, materialLibrary.addEntry(descriptor));
		}

		// Present
		{
			hknpMaterial material = materialLibrary.getEntry(hknpMaterialId::DEFAULT);
			material.m_name = "NamedMaterial";
			hknpMaterialId idA = materialLibrary.addEntry(material);
			hknpMaterialId idB = materialLibrary.addEntry(descriptor);
			HK_TEST(idA == idB);
			materialLibrary.removeEntry(idA);
		}
	}

	// ID descriptor
	{
		hknpMaterialDescriptor descriptor;

		// Not present
		{
			descriptor.m_materialId = hknpMaterialId(1);
			HK_TEST_ASSERT(0x623f8562, materialLibrary.addEntry(descriptor));
		}

		// Present
		{
			descriptor.m_materialId = hknpMaterialId::DEFAULT;
			hknpMaterialId id = materialLibrary.addEntry(descriptor);
			HK_TEST(id == hknpMaterialId::DEFAULT);
		}
	}

	// Reference material descriptor
	{
		hknpRefMaterial refMaterial;
		refMaterial.m_material = materialLibrary.getEntry(hknpMaterialId::DEFAULT);
		refMaterial.m_material.m_isExclusive = false;
		hknpMaterialDescriptor descriptor;
		descriptor.m_material = &refMaterial;

		// Not present
		hknpMaterialId idA;
		{
			idA = materialLibrary.addEntry(descriptor);
			HK_TEST(idA == hknpMaterialId(hknpMaterialId::NUM_PRESETS));
		}

		// Present
		{
			hknpMaterialId idB = materialLibrary.addEntry(descriptor);
			HK_TEST(idA == idB);
			materialLibrary.removeEntry(idA);
		}
	}
}

int NpUFMShapeTagCodec_main()
{
	testMaterialDescriptors();

	hkRefPtr<hknpShape> sphere;
	sphere.setAndDontIncrementRefCount(hknpSphereShape::createSphereShape(hkVector4::getZero(), 1));

	hknpMaterialLibrary library;
	hknpUFM358ShapeTagCodec codec(&library);

	// Create a material we can find by name
	hknpMaterialId namedMaterialId;
	{
		hknpMaterial material = library.getEntry(hknpMaterialId::DEFAULT);
		material.m_name = "NamedMaterial";
		material.m_isExclusive = false;
		namedMaterialId = library.addEntry(material);
	}

	// Create palette
	hknpMaterialPalette palette;
	palette.addEntry(hknpMaterialId::DEFAULT);
	palette.addEntry("NamedMaterial");
	hknpRefMaterial refMaterial;
	refMaterial.m_material = library.getEntry(hknpMaterialId::DEFAULT);
	refMaterial.m_material.m_isExclusive = false;
	palette.addEntry(&refMaterial);

	// Create geometry
	hkGeometry geometry;
	hkArray<hknpShapeTag> shapeTags;
	{
		hkArray<hkVector4>& vertices = geometry.m_vertices;
		vertices.setSize(5);
		vertices[0].set(-1, 0, -1);
		vertices[1].set(1, 0, -1);
		vertices[2].set(1, 0, 1);
		vertices[3].set(-1, 0, 1);
		vertices[4].set(0, 0, 0);

		shapeTags.setSize(3);
		shapeTags[0] = hknpUFM358ShapeTagCodec::encode(hknpMaterialPaletteEntryId(0), 2, 0);
		shapeTags[1] = hknpUFM358ShapeTagCodec::encode(hknpMaterialPaletteEntryId(1), 1, 1);
		shapeTags[2] = hknpUFM358ShapeTagCodec::encode(hknpMaterialPaletteEntryId(2), 0, 2);

		hkArray<hkGeometry::Triangle>& triangles = geometry.m_triangles;
		triangles.setSize(3);
		triangles[0].set(0, 4, 1, shapeTags[0]);
		triangles[1].set(1, 4, 2, shapeTags[1]);
		triangles[2].set(2, 4, 3, shapeTags[2]);
	}

	// Create shape
	hknpDefaultCompressedMeshShapeCinfo meshInfo(&geometry);
	hknpCompressedMeshShape cms(meshInfo);

	// Register shape in codec, creating in the process the palette materials that need to be created
	codec.registerShape(&cms, &palette);

	{
		// Check that the palette materials correspond to what we expect
		const hknpMaterialId* paletteMaterials;
		int numMaterials;
		codec.getPaletteMaterials(&palette, &paletteMaterials, &numMaterials);
		HK_TEST(numMaterials == 3);
		HK_TEST(paletteMaterials[0] == hknpMaterialId::DEFAULT);
		HK_TEST(paletteMaterials[1] == namedMaterialId);
		HK_TEST(paletteMaterials[2].value() == namedMaterialId.value() + 1);

		// Check decoded shape tags
		hknpUFM358ShapeTagCodec::Context context;
		context.m_parentShape = &cms;
		hkUint32 filterInfo = 0; hknpMaterialId materialId; hkUint64 userData = 0;
		for (int i = 0; i < shapeTags.getSize(); ++i)
		{
			codec.decode(shapeTags[i], &context, &filterInfo, &materialId, &userData);
			HK_TEST(filterInfo == hkUint32(shapeTags.getSize() - 1 - i));
			HK_TEST(materialId == paletteMaterials[i]);
			HK_TEST(userData == hkUlong(i));
		}
	}

	// Check decoding an unregistered shape
	{
		hkArray<hknpShapeInstance> instances(2);
		instances[0].setShape(sphere);
		instances[0].setShapeTag(HKNP_INVALID_SHAPE_TAG);
		instances[1].setShape(sphere);
		instances[1].setShapeTag(0);
		hknpStaticCompoundShape scs(instances.begin(), instances.getSize());

		hknpUFM358ShapeTagCodec::Context context;
		context.m_parentShape = &scs;
		hkUint32 filterInfo = 1;
		hknpMaterialId materialId = hknpMaterialId(1);
		hkUint64 userData = 1;

		// Invalid shape tags are not decoded
		codec.decode(instances[0].getShapeTag(), &context, &filterInfo, &materialId, &userData);
		HK_TEST(filterInfo == 1);
		HK_TEST(materialId == hknpMaterialId(1));
		HK_TEST(userData == 1);

		// Valid shape tags for unregistered shapes should override only collision filter info and user data
		codec.decode(instances[1].getShapeTag(), &context, &filterInfo, &materialId, &userData);
		HK_TEST(filterInfo == 0);
		HK_TEST(materialId == hknpMaterialId(1));
		HK_TEST(userData == 0);
	}

	// Check invalid context
	#if defined(HK_DEBUG)
	{
		// No parent shape
		hknpUFM358ShapeTagCodec::Context context;
		context.m_parentShape = HK_NULL;
		hkUint32 filterInfo; hknpMaterialId materialId; hkUint64 userData;
		HK_TEST_ASSERT(0x7df98ce6, codec.decode(0, &context, &filterInfo, &materialId, &userData));

		// Non-composite parent shape
		context.m_parentShape = sphere;
		HK_TEST_ASSERT(0x45d74d69, codec.decode(0, &context, &filterInfo, &materialId, &userData));
	}
	#endif

	// Check serialization
	{
		cms.setProperty( hknpShapePropertyKeys::MATERIAL_PALETTE, &palette );
		hkArray<char> buffer;
		hkOstream stream( buffer );
		hkSerializeUtil::save( &cms, stream.getStreamWriter() );
		hkResource* resource = hkSerializeUtil::load( buffer.begin(), buffer.getSize() );
		hknpCompressedMeshShape* loadedCms = resource->getContents<hknpCompressedMeshShape>();
		const hknpMaterialPalette* loadedPalette = (const hknpMaterialPalette*)loadedCms->getProperty(hknpShapePropertyKeys::MATERIAL_PALETTE);
		HK_TEST( loadedPalette != HK_NULL );
		codec.registerShape( loadedCms, loadedPalette );

		// Check decoded shape tags
		{
			// Read original palette materials again as their address may have changed after registering the loaded
			// palette
			const hknpMaterialId* paletteMaterials;
			int numPaletteMaterials;
			codec.getPaletteMaterials(&palette, &paletteMaterials, &numPaletteMaterials);

			const hknpMaterialId* loadedPaletteMaterials;
			int numLoadedPaletteMaterials;
			codec.getPaletteMaterials(loadedPalette, &loadedPaletteMaterials, &numLoadedPaletteMaterials);
			HK_TEST(numPaletteMaterials == numLoadedPaletteMaterials);
			HK_TEST(paletteMaterials[0] == loadedPaletteMaterials[0]);
			HK_TEST(paletteMaterials[1] == loadedPaletteMaterials[1]);
			HK_TEST(paletteMaterials[2] == loadedPaletteMaterials[2]);

			hknpUFM358ShapeTagCodec::Context context;
			context.m_parentShape = loadedCms;
			hkUint32 filterInfo = 0;
			hknpMaterialId materialId;
			hkUint64 userData = 0;
			for (int i = 0; i < shapeTags.getSize(); ++i)
			{
				codec.decode(shapeTags[i], &context, &filterInfo, &materialId, &userData);
				HK_TEST(filterInfo == hkUint32(shapeTags.getSize() - 1 - i));
				HK_TEST(materialId == loadedPaletteMaterials[i]);
				HK_TEST(userData == hkUlong(i));
			}
		}

		codec.unregisterShape(loadedCms);
		resource->removeReference();
	}

	// Test removing materials and invalidating palettes
	{
		// Create palette
		hknpMaterialPalette tempPalette;
		hknpRefMaterial tempRefMaterial;
		tempRefMaterial.m_material = library.getEntry(hknpMaterialId::DEFAULT);
		tempRefMaterial.m_material.m_name = "TempMaterial";
		hknpMaterialPaletteEntryId entryId = tempPalette.addEntry(&tempRefMaterial);

		// Create shape
		hkArray<hknpShapeInstance> instances(1);
		instances[0].setShape(sphere);
		instances[0].setShapeTag(codec.encode(entryId, 0, 0));
		hknpStaticCompoundShape scs(instances.begin(), instances.getSize());
		codec.registerShape(&scs, &tempPalette);

		// Removing a material in use by a palette should assert
		const hknpMaterialId* materials; int numMaterials;
		codec.getPaletteMaterials(&tempPalette, &materials, &numMaterials);
		hknpMaterialId materialId = materials[0];
		HK_TEST_ASSERT(0x6111911f, library.removeEntry(materialId));

		/// If there is no shape using the palette, the palette is invalidated and it should be fine to delete its
		// materials (the previous assert should have prevented the material from being removed)
		codec.unregisterShape(&scs);
		library.removeEntry(materialId);
	}

	// Test shape path decoding
	{
		hkArray<hknpShapeInstance> instances(1);
		instances[0].setShape(&cms);
		hknpStaticCompoundShape scs(instances.begin(), instances.getSize());

		hknpShapeCollectorWithInplaceTriangle leafShapeCollector;
		hkUint32 filterInfo; hknpMaterialId materialId; hkUint64 userData;
		hknpShapeTagCodec* codecPtr = &codec;
		int triangleIndex = 0;
		const hknpMaterialId* materials; int numMaterials;
		codec.getPaletteMaterials(&palette, &materials, &numMaterials);

		for( hkRefPtr<hknpShapeKeyIterator> it = scs.createShapeKeyIterator(); it->isValid(); it->next() )
		{
			leafShapeCollector.reset(hkTransform::getIdentity());
			scs.getLeafShape(it->getKey(), &leafShapeCollector);

			hknpUFM358ShapeTagCodec::Context context;
			context.m_parentShape = leafShapeCollector.m_parentShape;
			codecPtr->decode(leafShapeCollector.m_shapeTagPath.begin(), leafShapeCollector.m_shapeTagPath.getSize(),
							 leafShapeCollector.m_shapeTagOut, &context, &filterInfo, &materialId, &userData);

			// We are assuming here that triangles are obtained in the same order they appeared in the input geometry
			HK_TEST(filterInfo == hkUint32(shapeTags.getSize() - 1 - triangleIndex));
			HK_TEST(materialId == materials[triangleIndex]);
			HK_TEST(userData == hkUlong(triangleIndex));
			triangleIndex++;
		}
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(NpUFMShapeTagCodec_main, "Fast", "Physics/Test/UnitTest/Physics/", __FILE__);

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
