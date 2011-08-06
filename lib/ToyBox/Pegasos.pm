package ToyBox::Pegasos;

use strict;
use warnings;

use Data::Dumper;

our $VERSION = '0.0.1';

sub new {
    my $class = shift;
    my $self = {
        dnum => 0,
        data => [],
        lindex => {},
        rlindex => {},
        weight => {},
    } ;
    bless $self, $class;
}

sub add_instance {
    my ($self, %params) = @_;

    my $attributes = $params{attributes} or die "No params: attributes";
    my $label      = $params{label}     or die "No params: label";
    die "attributes is not hash ref"   unless ref($attributes) eq 'HASH';
    die "attributes is empty hash ref" unless keys %$attributes;
    $label = [$label] unless ref($label) eq 'ARRAY';

    my %copy_attr = %$attributes;

    foreach my $l (@$label) {
        if (! defined($self->{lindex}{$l})) {
            my $index = scalar(keys %{$self->{lindex}});
            $self->{lindex}{$l} = $index;
            $self->{rlindex}{$index} = $l;
        }
        my $datum = {f => \%copy_attr, l => $self->{lindex}{$l}};
        push(@{$self->{data}}, $datum);
        $self->{dnum}++;
    }
}

sub train {
    my ($self, %params) = @_;

    my $lambda = $params{lambda} || 1e-1;
    my $T      = $params{T}      || $self->{dnum};
    my $k      = $params{k}      || 16;
    die "lambda is le 0" unless scalar($lambda) > 0;
    die "T is le 0" unless scalar($T) > 0;
    die "k is le 0" unless scalar($k) > 0;

    my $progress_cb = $params{progress_cb};
    my $verbose = 0;
    $verbose = 1 if defined($progress_cb) && $progress_cb eq 'verbose';

    my $weight = $self->{weight};
    my $dnum   = $self->{dnum};
    my $scale_factors = {};
    $scale_factors->{$_} = 1 for keys %{$self->{rlindex}};

    print STDERR "learn start: " if $verbose;
    my $t = 0;
    while ($t < $T) {
        if ($verbose) {
            if    ($t % 1000 == 0) { print STDERR $t; }
            elsif ($t % 100  == 0) { print STDERR "."; }
        }
        $t += 1;
        my @data_array = ();
        foreach (1..$k) {
            push @data_array, $self->{data}[int(rand($dnum-1))];
        }

        my $eta = 1 / ($lambda * $t);

        my @scores_array = ();
        foreach my $datum (@data_array) {
            my $scores = {};
            while (my ($f, $val) = each %{$datum->{f}}) {
                next unless $weight->{$f};
                while (my ($l, $w) = each %{$weight->{$f}}) {
                    $scores->{$l} += $val * $scale_factors->{$l} * $w;
                }
            }
            push @scores_array, $scores;
        }

        if ($t > 1) {
            foreach my $l (keys %{$self->{rlindex}}) {
                $scale_factors->{$l} *= 1 - (1 / $t);
            }
        }

        foreach my $i (0..int(@data_array)-1) {
            my $datum  = $data_array[$i];
            my $scores = $scores_array[$i];
            foreach my $l (keys %{$self->{rlindex}}) {
                my $y = ($datum->{l} eq $l) ? 1 : -1;
                my $score = $scores->{$l} || 0;
                next if $y * $score > 1;

                my $coe = ($eta * $y) / ($k * $scale_factors->{$l});
                while (my ($f, $v) = each %{$datum->{f}}) {
                    $weight->{$f}{$l} += $coe * $v;
                }
            }
        }

        my $norms = {};
        foreach my $f (keys %$weight) {
            foreach my $l (keys %{$weight->{$f}}) {
                $norms->{$l} += $weight->{$f}{$l} ** 2;
            }
        }
        foreach my $l (keys %{$self->{rlindex}}) {
            next unless defined($norms->{$l});
            next unless $norms->{$l} > 0;
            my $tmp = 1 / ($scale_factors->{$l} * sqrt($lambda * $norms->{$l}));
            $scale_factors->{$l} *= $tmp if $tmp < 1;

            if ($scale_factors->{$l} < 1e-10) {
                foreach my $f (keys %$weight) {
                    foreach my $l2 (keys %{$weight->{$f}}) {
                        $weight->{$f}{$l2} *= $scale_factors->{$l2};
                    }
                }
                $scale_factors->{$_} = 1 for keys %{$self->{rlindex}};
            }
        }

    }
    foreach my $f (keys %$weight) {
        foreach my $l (keys %{$weight->{$f}}) {
            $weight->{$f}{$l} *= $scale_factors->{$l};
        }
    }
    $self->{weight} = $weight;
    print STDERR "done($t)\n" if $verbose;

    1;
}

sub predict {
    my ($self, %params) = @_;

    my $attributes = $params{attributes} or die "No params: attributes";
    die "attributes is not hash ref"   unless ref($attributes) eq 'HASH';
    die "attributes is empty hash ref" unless keys %$attributes;

    my $weight = $self->{weight};
    my $rlindex = $self->{rlindex};

    my $scores = {};
    while (my ($f, $val) = each %$attributes) {
        while (my ($l, $w) = each %{$weight->{$f}}) {
            $scores->{$rlindex->{$l}} += $val * $w;
        }
    }

    $scores;
}

sub labels {
    my $self = shift;
    keys %{$self->{lindex}};
}

sub data_num {
    my $self = shift;
    $self->{dnum};
}

1;
__END__


=head1 NAME

ToyBox::Pegasos - Classifier using Pegasos Algorithm

=head1 SYNOPSIS

  use ToyBox::Pegasos;

  my $svm= ToyBox::Pegasos->new();
  
  $svm->add_instance(
      attributes => {a => 2, b => 3},
      label => 'positive'
  );
  
  $svm->add_instance(
      attributes => {c => 3, d => 1},
      label => 'negative'
  );
  
  $svm->train(T => 200,
              lambda => 1.0,
              k => 16,
              progress_cb => 'verbose');
  
  my $score = $svm->predict(
                  attributes => {a => 1, b => 1, d => 1, e =>1}
              );

=head1 DESCRIPTION

=head1 AUTHOR

TAGAMI Yukihiro <tagami.yukihiro@gmail.com>

=head1 LICENSE

This library is distributed under the term of the MIT license.

L<http://opensource.org/licenses/mit-license.php>

