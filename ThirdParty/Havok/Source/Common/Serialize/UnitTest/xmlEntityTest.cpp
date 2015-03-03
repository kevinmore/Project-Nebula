/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/System/Io/IStream/hkIStream.h>
#include <Common/Base/System/Io/Reader/Memory/hkMemoryStreamReader.h>
#include <Common/Serialize/Serialize/hkRelocationInfo.h>
#include <Common/Serialize/Serialize/Xml/hkXmlObjectReader.h>
#include <Common/Serialize/Serialize/Xml/hkXmlObjectWriter.h>
#include <Common/Serialize/Util/hkRootLevelContainer.h>
#include <Common/Serialize/Util/Xml/hkXmlParser.h>

static const char xmlEntityTest_asciiData[] = "This\tis <t\x7fth\x7fhe\177e> ori\001ginal d\020 \xfata & 'stuff'.";

static void xmlEntityTest_main()
{
	extern const class hkClass hkRootLevelContainerNamedVariantClass;
	hkArray<char> xmlOutput;
	{
		hkOstream out(xmlOutput);
		hkXmlObjectWriter::SequentialNameFromAddress namer;
		hkXmlObjectWriter writer(namer);
		hkRootLevelContainer::NamedVariant named(const_cast<char*>(xmlEntityTest_asciiData), HK_NULL, HK_NULL );
		hkResult res = writer.writeObjectWithElement( out.getStreamWriter(), &named, hkRootLevelContainerNamedVariantClass, HK_NULL );
		HK_TEST( res == HK_SUCCESS );
	}

	{
		hkIstream inStream(xmlOutput.begin(), xmlOutput.getSize());
		hkArray<char> xmlInput;
		hkXmlParser parser;
		hkStringMap<void*> nameToObject; // map of string id to object address
		hkXmlObjectReader reader(&parser, &nameToObject);
		hkRelocationInfo reloc;
		hkResult res = reader.readObject( inStream.getStreamReader(), xmlInput, hkRootLevelContainerNamedVariantClass, reloc );
		reloc.applyLocalAndGlobal( xmlInput.begin() );
		hkRootLevelContainer::NamedVariant* named = (hkRootLevelContainer::NamedVariant*)xmlInput.begin();

		HK_TEST( res == HK_SUCCESS );
		HK_TEST( hkString::strLen(named->getName()) == sizeof(xmlEntityTest_asciiData)-1 );
		HK_TEST( hkString::memCmp(xmlEntityTest_asciiData, named->getName(), sizeof(xmlEntityTest_asciiData)-1 ) == 0 );
	}
}

static void xmlparser_test()
{
	const char xml[] = 
		"<hkparam name=\"blahhhh\">\n"
		"	<struct>\n"
		"		<hkparam name=\"blahhhh1\">0</hkparam>\n"
		"		<hkparam name=\"blahhhh1\">0</hkparam>\n"
		"		<hkparam name=\"blahhhh1\">0</hkparam>\n"
		"		<hkparam name=\"blahhhh1\">0</hkparam>\n"
		"		<hkparam name=\"blahhhh1\">0</hkparam>\n"
		"	</struct>\n"
		"</hkparam>";
	hkXmlParser parser;
	hkMemoryStreamReader sr(xml, HK_COUNT_OF(xml), hkMemoryStreamReader::MEMORY_INPLACE);
	hkXmlParser::Node* root;
	parser.nextNode(&root, &sr);
	if( hkXmlParser::StartElement* start = root->asStart())
	{
		hkTree< hkRefPtr<hkXmlParser::Node> > xmltree;
		parser.expandNode(start, xmltree, &sr);
	}
	else
	{
		HK_TEST(0);
	}
	
	root->removeReference();
}

int xmlTest_main()
{
	xmlparser_test();
	xmlEntityTest_main();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(xmlTest_main, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__     );

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
