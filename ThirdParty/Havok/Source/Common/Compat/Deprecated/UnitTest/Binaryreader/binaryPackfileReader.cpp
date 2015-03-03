/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/hkSerialize.h>
#include <Common/Base/Types/hkIgnoreDeprecated.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Reflection/Registry/hkTypeInfoRegistry.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>
#include <Common/Base/System/Io/IStream/hkIStream.h>
#include <Common/Compat/Deprecated/Packfile/Binary/hkBinaryPackfileReader.h>
#include <Common/Serialize/Packfile/Binary/hkBinaryPackfileWriter.h>
#include <Common/Serialize/Packfile/Binary/hkPackfileSectionHeader.h>
#include <Common/Compat/Deprecated/UnitTest/Binaryreader/hkSomeObject.h>

int hkSomeObject::m_numInstances;

extern const hkTypeInfo hkSomeObjectTypeInfo;
extern const hkTypeInfo hkClassTypeInfo;
extern const hkTypeInfo hkClassMemberTypeInfo;
extern const hkTypeInfo hkClassEnumTypeInfo;

static int binaryPackfileReader()
{
	hkArray<char> buf;

	const char sectionName[] = "second";
	{
		hkVtableClassRegistry localVtableReg;
		localVtableReg.registerVtable(hkSomeObjectTypeInfo.getVtable(), &hkSomeObjectClass);

		HK_TEST( hkSomeObject::m_numInstances == 0 );
		hkSomeObject obj3;
		hkSomeObject obj2(&obj3);
		hkSomeObject obj1(&obj2);

		hkBinaryPackfileWriter writer;
		writer.addSection(sectionName);
		writer.setSectionForPointer(&obj2, sectionName);

		writer.setContentsWithRegistry( &obj1, hkSomeObjectClass, &localVtableReg );

		writer.addExport(&obj1, "ex1");
		writer.addExport(&obj2, "exported2");
		writer.addImport(&obj3, "i");

		hkOstream outfile( buf );
		hkPackfileWriter::Options options;
		writer.save( outfile.getStreamWriter(), options );
		HK_TEST( hkSomeObject::m_numInstances == 3 );
	}
	{
		hkTypeInfoRegistry localTypeInfoReg;
		// default classes
		localTypeInfoReg.registerTypeInfo(&hkClassTypeInfo);
		localTypeInfoReg.registerTypeInfo(&hkClassMemberTypeInfo);
		localTypeInfoReg.registerTypeInfo(&hkClassEnumTypeInfo);
		// unit test class
		localTypeInfoReg.registerTypeInfo(&hkSomeObjectTypeInfo);

		

		hkBinaryPackfileReader reader;
		hkIstream infile(buf.begin(), buf.getSize());
		reader.loadEntireFile( infile.getStreamReader() );
		HK_TEST( hkSomeObject::m_numInstances == 0 );

		hkError::getInstance().setEnabled(0x38afeb70, false);
		hkSomeObject* root = (hkSomeObject*)reader.getContentsWithRegistry("hkSomeObject", &localTypeInfoReg);
		hkError::getInstance().setEnabled(0x38afeb70, true);

		HK_TEST( hkSomeObject::m_numInstances == 2 );
		HK_TEST( root->m_next != HK_NULL );

		hkArray<hkResource::Export> exports0;
		hkArray<hkResource::Import> imports0;
		reader.getPackfileData()->getImportsExports( imports0, exports0 );
		HK_TEST( exports0.getSize() == 2 );
		HK_TEST( imports0.getSize() == 1 );

		int sectionIndex = 0;
		for( ; sectionIndex < reader.getNumSections(); ++sectionIndex )
		{
			if( hkString::strCmp(reader.getSectionHeader(sectionIndex).m_sectionTag, sectionName ) == 0 )
			{
				reader.unloadSection( sectionIndex );
				reader.fixupGlobalReferences();
				break;
			}
		}
		HK_TEST( sectionIndex < reader.getNumSections() );

		hkArray<hkResource::Export> exports1;
		hkArray<hkResource::Import> imports1;
		reader.getPackfileData()->getImportsExports( imports1, exports1 );
		HK_TEST( exports1.getSize() == 1 );
		HK_TEST( imports1.getSize() == 0 );

		HK_TEST( hkSomeObject::m_numInstances == 1 );
		HK_TEST( root->m_next == HK_NULL );
	}
	HK_TEST( hkSomeObject::m_numInstances == 0 );
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(binaryPackfileReader, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__     );

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
