use inc::Module::Install;

name          'ToyBox-SVMSGD';
all_from      'lib/ToyBox/SVMSGD.pm';
license       'MIT';

test_requires 'Test::More';

WriteAll;
