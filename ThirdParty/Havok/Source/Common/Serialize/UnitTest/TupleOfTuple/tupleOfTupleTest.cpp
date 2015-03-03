/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Base/System/Io/IStream/hkIStream.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Serialize/Serialize/Xml/hkXmlObjectReader.h>
#include <Common/Serialize/Serialize/Xml/hkXmlObjectWriter.h>
#include <Common/Serialize/Data/Native/hkDataObjectNative.h>

#include <Common/Serialize/Util/hkSerializeUtil.h>

#include <Common/Base/Reflection/Registry/hkDynamicClassNameRegistry.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>

#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>

#include <Common/Serialize/UnitTest/TupleOfTuple/tupleOfTupleTest.h>
#include <Common/Serialize/Tagfile/Binary/hkBinaryTagfileWriter.h>

extern const hkTypeInfo hkReferencedObjectTypeInfo;
extern const hkClass hkReferencedObjectClass;

extern const hkTypeInfo hkBaseObjectTypeInfo;
extern const hkClass hkBaseObjectClass;

/* Check we can have arrays/tuples of structs containing arrays/tuples of structs */

static const hkClass* classes[] =
{
	&hkBaseObjectClass,
	&hkReferencedObjectClass, 
	&TupleOfTupleClass,
	&TupleOfTupleChildClass,
	&TupleOfTupleChildChildChildClass
};

static const hkTypeInfo* infos[] =
{
	&hkReferencedObjectTypeInfo,
	&hkBaseObjectTypeInfo,
	&TupleOfTupleTypeInfo,
	&TupleOfTupleChildTypeInfo,
	&TupleOfTupleChildChildChildTypeInfo,
};

int tupleOfTupleTest_main()
{
	hkDynamicClassNameRegistry classReg;
	hkTypeInfoRegistry typeReg;
	hkVtableClassRegistry vtableReg;

	for( unsigned int i = 0; i < HK_COUNT_OF(classes); ++i )
	{
		const hkClass* cls = classes[i];
		const hkTypeInfo* info = infos[i];

		classReg.registerClass(cls);
		if( cls->hasVtable() )
		{
			vtableReg.registerVtable( info->getVtable(), cls );
		}
		
		typeReg.registerTypeInfo(info);
	}

	hkDataWorldNative nativeWorld;

	nativeWorld.setClassRegistry(&classReg);
	nativeWorld.setVtableRegistry(&vtableReg);

	// Class needs to be registered

	TupleOfTuple tuples;

	hkPseudoRandomGenerator rand(1000);

	{
		int num0 = (rand.getRand32() % 100) + 10;
		for (int i = 0; i < num0; i++)
		{
			TupleOfTuple::Child& child = tuples.m_children.expandOne();

			int num1 = (rand.getRand32() % 10) + 4;
			for (int j = 0; j < num1; j++)
			{
				TupleOfTuple::Child::ChildChild& childChild = child.m_children.expandOne();

				int num2 = (rand.getRand32() % 7);
				childChild.m_nums.setSize(num2);
				for (int k = 0; k < num2; k++)
				{
					childChild.m_nums[k] = (int)rand.getRand32();
				}

				hkStringBuf buf;
				buf.appendPrintf("Hello %i %i", i, j);
				childChild.m_text = buf;
			}
		}
	}

	{
		const int size0 = HK_COUNT_OF(tuples.m_tuple);
		for (int i = 0; i < size0; i++)
		{
			TupleOfTuple::Child& child = tuples.m_tuple[i];
			const int size1 = HK_COUNT_OF(child.m_tuple);
			for (int j = 0; j < size1; j++)
			{
				TupleOfTuple::Child::ChildChild& childChild = child.m_tuple[j];
				const int size2 = HK_COUNT_OF(childChild.m_tuple);

				for (int k = 0; k < size2; k++)
				{
					childChild.m_tuple[k] = (int)rand.getRand32();
				}
			}
		}
	}

	hkArray<char> buffer;
	
	// Save it off
	{
		hkOstream stream(buffer);

		hkDataObject obj = nativeWorld.wrapObject(&tuples, TupleOfTupleClass);
		HK_ON_DEBUG(hkResult res =) hkBinaryTagfileWriter().save(obj, stream.getStreamWriter(), HK_NULL);
		HK_ASSERT(0x32423423, res == HK_SUCCESS);
	}

	// Read it back in again
	hkRefPtr<hkResource> resource;
	{
		hkSerializeUtil::LoadOptions options;
		options.useClassNameRegistry(&classReg);
		options.useTypeInfoRegistry(&typeReg);

		resource.setAndDontIncrementRefCount( hkSerializeUtil::load(buffer.begin(), buffer.getSize(), HK_NULL, options) );
	}

	HK_ASSERT(0x3464565a, resource);

	// Get the contents
	TupleOfTuple* readTuples = resource->getContentsWithRegistry<TupleOfTuple>(&typeReg);
	HK_ASSERT2(0xa6451543, readTuples != HK_NULL, "Could not load root level obejct" );

	HK_TEST(tuples == *readTuples);

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(tupleOfTupleTest_main, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__     );

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
